from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from homr.transformer.configs import Config
from homr.transformer.structured_decode import decode_note
from homr.transformer.vocabulary import (
    EncodedSymbol,
    has_rhythm_symbol_a_position,
    kern_to_symbol_duration,
)
from training.architecture.transformer.custom_x_transformer import (
    AbsolutePositionalEmbedding,
    AttentionLayers,
    CustomDecoder,
    Intermediates,
    LayerIntermediates,
    TokenEmbedding,
)
from training.architecture.transformer.profile_context import ProfileContextEmbedding
from training.architecture.transformer.staff_context import StaffContextTransformer
from training.architecture.transformer.structured_heads import StructuredNotationHeads


class ScoreTransformerWrapper(nn.Module):
    """
    Based on x_transformers.TransformerWrapper to support multiple embeddings.
    """

    def __init__(
        self,
        config: Config,
        attn_layers: Any,
        l2norm_embed: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(attn_layers, AttentionLayers):
            raise ValueError("attention layers must be an instance of AttentionLayers")

        dim = attn_layers.dim
        self.max_seq_len = config.max_seq_len
        self.l2norm_embed = l2norm_embed
        self.lift_emb = TokenEmbedding(
            config.decoder_dim, config.num_lift_tokens, l2norm_embed=l2norm_embed
        )
        self.pitch_emb = TokenEmbedding(
            config.decoder_dim, config.num_pitch_tokens, l2norm_embed=l2norm_embed
        )
        self.rhythm_emb = TokenEmbedding(
            config.decoder_dim, config.num_rhythm_tokens, l2norm_embed=l2norm_embed
        )
        self.articulation_emb = TokenEmbedding(
            config.decoder_dim, config.num_articulation_tokens, l2norm_embed=l2norm_embed
        )
        self.slur_emb = TokenEmbedding(
            config.decoder_dim, config.num_slur_tokens, l2norm_embed=l2norm_embed
        )

        self.pos_emb = AbsolutePositionalEmbedding(
            config.decoder_dim, config.max_seq_len, l2norm_embed=l2norm_embed
        )
        self.attention_dim = config.max_width * config.max_height // config.patch_size**2 + 1
        self.attention_width = config.max_width // config.patch_size
        self.attention_height = config.max_height // config.patch_size
        self.patch_size = config.patch_size

        self.attn_layers = attn_layers
        self.post_emb_norm = nn.LayerNorm(dim)

        self.to_logits_lift = nn.Linear(dim, config.num_lift_tokens)
        self.to_logits_pitch = nn.Linear(dim, config.num_pitch_tokens)
        self.to_logits_rhythm = nn.Linear(dim, config.num_rhythm_tokens)
        self.to_logits_position = nn.Linear(dim, config.num_position_tokens)
        self.to_logits_articulations = nn.Linear(dim, config.num_articulation_tokens)
        self.to_logits_slurs = nn.Linear(dim, config.num_slur_tokens)
        self.init_()

    def init_(self) -> None:
        if self.l2norm_embed:
            nn.init.normal_(self.lift_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pitch_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.rhythm_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.articulation_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.slur_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
        else:
            # Use transformer standard initialization (std=0.02)
            # This provides stronger gradients than 1e-5 for faster convergence
            nn.init.normal_(self.lift_emb.emb.weight, std=0.02)
            nn.init.normal_(self.pitch_emb.emb.weight, std=0.02)
            nn.init.normal_(self.rhythm_emb.emb.weight, std=0.02)
            nn.init.normal_(self.articulation_emb.emb.weight, std=0.02)
            nn.init.normal_(self.slur_emb.emb.weight, std=0.02)
            nn.init.normal_(self.pos_emb.emb.weight, std=0.02)

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        slurs: torch.Tensor,
        context: torch.Tensor | None = None,
        cache_len: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_center_of_attention: bool = False,
        profile_context_emb: torch.Tensor | None = None,
        staff_context_emb: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> Any:
        # §7.2/§7.3's score-profile conditioning: an already-computed, per-sequence
        # additive vector (training.architecture.transformer.profile_context.
        # ProfileContextEmbedding's output, shape (batch, dim)) - this class stays
        # decoupled from how it was built, the same way it does not know how the
        # rhythm/pitch/... embeddings' vocabularies were chosen. Broadcasts across every
        # position via unsqueeze(1), since profile context does not vary within one
        # staff's decode. `None` (the default) leaves `x` exactly as it was before this
        # parameter existed - no caller that omits it is affected in any way.
        profile_bias = (
            profile_context_emb.unsqueeze(1) if profile_context_emb is not None else None
        )
        # §4/§7.4 Stage C: the same shape and broadcast rule as profile_bias above, but
        # a genuinely separate signal (this staff's own learned cross-staff context,
        # StaffContextTransformer's gated output for this staff, computed from a first
        # decode pass) - kept as its own parameter rather than folded into
        # profile_context_emb so the two conditioning sources stay independently
        # ablatable, not conflated into one. `None` leaves `x` exactly as it was
        # before this parameter existed.
        staff_bias = (
            staff_context_emb.unsqueeze(1) if staff_context_emb is not None else None
        )
        cache = kwargs.pop("cache", None)
        if cache is None:
            x = (
                self.rhythm_emb(rhythms)
                + self.pitch_emb(pitchs)
                + self.lift_emb(lifts)
                + self.articulation_emb(articulations)
                + self.slur_emb(slurs)
                + self.pos_emb(rhythms)
            )
            if profile_bias is not None:
                x = x + profile_bias
            if staff_bias is not None:
                x = x + staff_bias

            x = self.post_emb_norm(x)

            if return_center_of_attention:
                x, hiddens = self.attn_layers(x, mask=mask, return_hiddens=True, **kwargs)
                attention = self.get_center_of_attention(hiddens.attn_intermediates)
            else:
                x = self.attn_layers(x, mask=mask, return_hiddens=False, context=context, **kwargs)
                attention = None

            out_lifts = self.to_logits_lift(x)
            out_pitchs = self.to_logits_pitch(x)
            out_rhythms = self.to_logits_rhythm(x)
            out_articulations = self.to_logits_articulations(x)
            out_slurs = self.to_logits_slurs(x)
            out_positions = self.to_logits_position(x)
            return (
                out_rhythms,
                out_pitchs,
                out_lifts,
                out_positions,
                out_articulations,
                out_slurs,
                x,
                attention,
                None,
            )

        else:
            x = (
                self.rhythm_emb(rhythms)
                + self.pitch_emb(pitchs)
                + self.lift_emb(lifts)
                + self.articulation_emb(articulations)
                + self.slur_emb(slurs)
                + self.pos_emb(rhythms, offset=cache_len)
            )
            if profile_bias is not None:
                x = x + profile_bias
            if staff_bias is not None:
                x = x + staff_bias

            x = self.post_emb_norm(x)

            # reconstruct x_transformers LayerIntermediates from the input_cache
            inters = []
            for i in range(0, 32, 2):
                inters.append(Intermediates(cached_kv=(cache[i], cache[i + 1])))

            cache_input = LayerIntermediates(attn_intermediates=inters, cache_length=cache_len)

            x, cache = self.attn_layers(
                x, cache=cache_input, mask=mask, return_hiddens=True, context=context
            )

            # get the kv cache tensors from the LayerIntermediates class
            # the cache is built up like this:
            # LayerIntermediates(atten_intermediates=Intermediates(cached_kv=(cache_k, cache_v)))
            # cache is alternating between shapes (batch, 8, seq_len, 64) and (batch, 8, 1281, 64)
            # 8 probably corresponds to the number of decoder_heads
            # 1281 is the same as the encoder output
            cache_out = []
            attn_inters = cache.attn_intermediates
            if return_center_of_attention:
                attention = self.get_center_of_attention(attn_inters)
            else:
                attention = None
            for i in range(16):  # 16x2
                k, v = attn_inters[i].cached_kv
                cache_out.append(k)
                cache_out.append(v)

            out_lifts = self.to_logits_lift(x)
            out_pitchs = self.to_logits_pitch(x)
            out_rhythms = self.to_logits_rhythm(x)
            out_articulations = self.to_logits_articulations(x)
            out_slurs = self.to_logits_slurs(x)
            out_positions = self.to_logits_position(x)
            return (
                out_rhythms,
                out_pitchs,
                out_lifts,
                out_positions,
                out_articulations,
                out_slurs,
                x,
                attention,
                cache_out,
            )

    def get_center_of_attention(self, intermediates: list[Any]) -> torch.Tensor:
        """
        Calculates the center of attention. It uses the attention weights,
        performs a power scaling to give more weight to the peaks and then
        calculates the center of mass to get the focus point of the attention.
        """

        # Only use the last 3 layers as the later layers contain the strongest
        # alignment between attention and semantic object position.
        filtered_intermediate = [
            intermediates[-5].post_softmax_attn[:, :, -1, :],
            intermediates[-3].post_softmax_attn[:, :, -1, :],
            intermediates[-1].post_softmax_attn[:, :, -1, :],
        ]

        attention_all_layers = torch.mean(torch.stack(filtered_intermediate), dim=0)
        attention_all_layers = attention_all_layers.squeeze(0).squeeze(1)
        attention_all_layers = attention_all_layers.mean(dim=0)
        h, w = self.attention_height, self.attention_width

        image_token_count = h * w
        image_attention = attention_all_layers[0:image_token_count]

        image_attention_2d = image_attention.reshape(h, w)

        power = 4.0
        weights = torch.clamp(image_attention_2d, min=1e-4).pow(power)

        y_coords = torch.linspace(0.5, h - 0.5, h, device=weights.device, dtype=weights.dtype)
        x_coords = torch.linspace(0.5, w - 0.5, w, device=weights.device, dtype=weights.dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        total_mass = weights.sum()
        row = (weights * yy).sum() / total_mass
        col = (weights * xx).sum() / total_mass

        center_of_attention = torch.stack(
            [
                col * self.patch_size,
                row * self.patch_size,
            ]
        )

        return center_of_attention


#: context_to_batch_fields' own keys (profile_context.py) - what DataLoader.__getitem__
#: emits per sample and the default collator stacks into a batch. Named here, not
#: imported from profile_context.py, since ScoreDecoder needs to pop exactly these keys
#: out of **kwargs regardless of whether ProfileContextEmbedding is even attached.
_PROFILE_BATCH_FIELDS = (
    "profile_present",
    "profile_family_index",
    "profile_part_ordinal_index",
    "profile_staff_within_part_index",
    "profile_staff_count_index",
    "profile_clef_indices",
    "profile_clef_count",
    "profile_transposition_index",
    "profile_time_signature_index",
)


class ScoreDecoder(nn.Module):
    def __init__(self, transformer: ScoreTransformerWrapper, config: Config):
        super().__init__()
        self.pad_value = (config.pad_token,)
        # NOTE: nonote_token and pad_token currently share the same id (0).
        # So we cannot use token id as ignore_index, otherwise "nonote" labels are ignored.
        self.ignore_index = -100
        self.config = config
        self.net = transformer
        self.max_seq_len = config.max_seq_len
        self.eos_token = config.eos_token

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}
        self.inv_articulation_vocab = {v: k for k, v in config.articulation_vocab.items()}
        self.inv_slur_vocab = {v: k for k, v in config.slur_vocab.items()}
        self.inv_position_vocab = {v: k for k, v in config.position_vocab.items()}

        note_mask = torch.zeros(config.num_rhythm_tokens)
        for index, rhythm_symbol in enumerate(config.rhythm_vocab.keys()):
            if has_rhythm_symbol_a_position(rhythm_symbol):
                note_mask[index] = 1
        self.note_mask = nn.Parameter(note_mask)

        # DECODER_RHYTHM_ACCURACY_DESIGN.md §7.3's ground-truth-supervised duration-
        # adherence loss (calDurationAdherenceLoss below): a fixed, non-trainable
        # per-token lookup of each rhythm token's real duration (whole-note units,
        # `kern_to_symbol_duration` - the exact same parser `EncodedSymbol.get_duration`
        # itself calls). Tokens that don't start with "note"/"rest" (barline, clef,
        # keySignature, "chord") correctly get 0 here.
        #
        # "chord" itself is a bare marker (0 duration, correct) - but a chord's *later*
        # members are NOT the "chord" token, they carry their own real note_/rest_
        # token immediately after it (`training/omr_datasets/staff_merging.py`'s
        # `create_chord_over_two_staffs`: `if not is_first: result.append(chord
        # marker); result.append(symbol)` - the marker is inserted *before* each
        # non-first member, never replacing it). An earlier version of this comment
        # claimed summing this vector position-by-position was already correct because
        # of that; it was wrong, and so was the loss below until this fix - confirmed
        # directly: a real two-note simultaneous quarter-note chord (`note_4, chord,
        # note_4`) sums to 0.5 here, double the true 0.25, because both real note
        # tokens contribute independently. `rhythm_is_chord_continuation` below is
        # what corrects this: it flags exactly the positions that immediately follow a
        # "chord" marker, so both loss methods can zero those positions out of the
        # cumulative sum instead of double-counting them - the same *grouping*
        # `homr.music_xml_generator.group_into_chords`/`SymbolChord.get_duration`
        # already do elsewhere in this codebase, simplified to "count the first member,
        # not the minimum across members" (equivalent for correctly-notated chords,
        # whose simultaneous members always share one duration by definition - the
        # minimum only differs from the first for a decode already wrong in a way this
        # loss cannot represent anyway) rather than a second recurrent-grouping
        # implementation inside a batched, differentiable loss.
        duration_vec = torch.zeros(config.num_rhythm_tokens)
        for index, rhythm_symbol in enumerate(config.rhythm_vocab.keys()):
            if rhythm_symbol.startswith(("note", "rest")):
                kern = rhythm_symbol.split("_")[1]
                duration_vec[index] = float(kern_to_symbol_duration(kern).fraction)
        # A non-persistent buffer, not an nn.Parameter like note_mask above: this is a
        # fixed, deterministic lookup table with nothing to learn or load - a
        # persistent (state_dict-included) registration would make every existing
        # pinned checkpoint (saved before this field existed) look like it's "missing"
        # a real parameter to load_checkpoint's mismatch check, when there is nothing
        # to load in the first place.
        self.register_buffer("rhythm_duration", duration_vec, persistent=False)

        # Same "barline" definition `SymbolChord.is_barline` already uses elsewhere in
        # this codebase (cross_staff_consistency.py's own cumulative-position checks
        # rely on it too) - a repeat barline closes a measure exactly as an ordinary
        # one does.
        barline_mask = torch.zeros(config.num_rhythm_tokens)
        for index, rhythm_symbol in enumerate(config.rhythm_vocab.keys()):
            if "barline" in rhythm_symbol or "repeat" in rhythm_symbol:
                barline_mask[index] = 1
        self.register_buffer("rhythm_is_barline", barline_mask, persistent=False)

        # Marks the single literal "chord" token itself (see the long comment on
        # `rhythm_duration` above) - both duration losses shift this by one position to
        # find which positions are a chord's non-first members, not this token.
        chord_marker_mask = torch.zeros(config.num_rhythm_tokens)
        for index, rhythm_symbol in enumerate(config.rhythm_vocab.keys()):
            if rhythm_symbol == "chord":
                chord_marker_mask[index] = 1
        self.register_buffer("rhythm_is_chord_marker", chord_marker_mask, persistent=False)

        # Output-only heads over the shared hidden state. They add dictionary keys to
        # forward's result and nothing else - the existing losses and the positional
        # tuple ScoreTransformerWrapper returns are untouched, so a run without them is
        # bit-identical to before.
        self.structured_heads = (
            StructuredNotationHeads(
                dim=config.decoder_dim,
                beam_levels=config.structured_beam_levels,
                slur_slots=config.structured_slur_slots,
            )
            if getattr(config, "enable_structured_heads", False)
            else None
        )

        # §7.2/§7.3 score-profile conditioning - same off-by-default reasoning as
        # structured_heads above, and its own zero-initialized gate means enabling it
        # is still a no-op until training moves the gate (profile_context.py).
        self.profile_context = (
            ProfileContextEmbedding(dim=config.decoder_dim)
            if getattr(config, "enable_profile_context", False)
            else None
        )

        # §4/§7.4 Stage C: learned cross-staff coherence. Same off-by-default and
        # zero-init-gate reasoning as profile_context above - a checkpoint trained
        # without it keeps loading and behaving exactly as before, and enabling it is
        # itself still a no-op until training moves the gate (train_staff_context.py).
        self.staff_context = (
            StaffContextTransformer(dim=config.decoder_dim)
            if getattr(config, "enable_staff_context", False)
            else None
        )

        # Weight the actual lift tokens (so neither nonote nor null) higher
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _pop_profile_context_emb(self, kwargs: dict[str, Any]) -> torch.Tensor | None:
        """Pop this batch's `profile_*` index tensors (`context_to_batch_fields`'s own
        keys, stacked by the default collator) out of `kwargs` and reduce them to one
        additive vector - before `kwargs` propagates any further, since
        `ScoreTransformerWrapper`'s own `**kwargs` passes through to `attn_layers`,
        which has no idea what to do with raw profile index tensors. Popped
        unconditionally, whether or not `self.profile_context` is attached - a
        checkpoint with `enable_profile_context=False` must not choke on a batch that
        happens to carry these keys (mixed-corpus training, or a caller upgraded the
        dataloader before the config).
        """
        present = {key: kwargs.pop(key, None) for key in _PROFILE_BATCH_FIELDS}
        if self.profile_context is None or any(value is None for value in present.values()):
            return None
        return self.profile_context.forward_from_batch(**present)

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        nonote_tokens: torch.Tensor,
        **kwargs: Any,
    ) -> list[EncodedSymbol]:
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out_rhythm = start_tokens
        out_pitch = nonote_tokens
        out_lift = nonote_tokens
        out_articulations = nonote_tokens
        out_slurs = nonote_tokens
        mask = kwargs.pop("mask", None)
        context_first = kwargs.pop("context")
        context_later = context_first[:, :0]

        if mask is None:
            # the mask is always (True, True) because the x_ are always (1, 1)
            # and contain only the last token
            # the information about the rest of the tokens gets passed via the kv cache
            mask = torch.ones((1, 1), dtype=torch.bool, device=self.device)

        symbols: list[EncodedSymbol] = []

        cache = init_cache(0, self.device)[0]

        for step in range(self.max_seq_len):
            x_lift = out_lift[:, -1:]
            x_pitch = out_pitch[:, -1:]
            x_rhythm = out_rhythm[:, -1:]
            x_articulations = out_articulations[:, -1:]
            x_slurs = out_slurs[:, -1:]

            if step == 0:
                context = context_first
            else:
                context = context_later

            (
                rhythmsp,
                pitchsp,
                liftsp,
                positionsp,
                articulationsp,
                slursp,
                hidden,
                _,
                cache,
            ) = self.net(
                rhythms=x_rhythm,
                pitchs=x_pitch,
                lifts=x_lift,
                articulations=x_articulations,
                slurs=x_slurs,
                context=context,
                cache_len=torch.Tensor([step]).to(self.device).long(),
                mask=mask,
                cache=cache,
                return_center_of_attention=False,
                **kwargs,
            )

            # Greedy decoding: pick the highest logit directly for each output
            rhythm_sample = rhythmsp[:, -1, :].argmax(dim=-1, keepdim=True)
            pitch_sample = pitchsp[:, -1, :].argmax(dim=-1, keepdim=True)
            lift_sample = liftsp[:, -1, :].argmax(dim=-1, keepdim=True)
            articulation_sample = articulationsp[:, -1, :].argmax(dim=-1, keepdim=True)
            slur_sample = slursp[:, -1, :].argmax(dim=-1, keepdim=True)
            position_sample = positionsp[:, -1, :].argmax(dim=-1, keepdim=True)

            lift_token = detokenize(lift_sample, self.inv_lift_vocab)
            pitch_token = detokenize(pitch_sample, self.inv_pitch_vocab)
            rhythm_token = detokenize(rhythm_sample, self.inv_rhythm_vocab)
            articulation_token = detokenize(articulation_sample, self.inv_articulation_vocab)
            slur_token = detokenize(slur_sample, self.inv_slur_vocab)
            position_token = detokenize(position_sample, self.inv_position_vocab)

            if rhythm_sample[0][0] == self.eos_token:
                break

            symbol = EncodedSymbol(
                rhythm=rhythm_token[0],
                pitch=pitch_token[0],
                lift=lift_token[0],
                articulation=articulation_token[0],
                slur=slur_token[0],
                position=position_token[0],
            )
            # The structured heads run on the same hidden state the branch logits came
            # from, for this step's token only. Without this the heads are trained,
            # evaluated and never used: `notation` stays None and `build_beams` returns
            # early on every note, so scores carry MuseScore's auto-beaming instead of
            # the predicted beams.
            if self.structured_heads is not None:
                head_logits = self.structured_heads(hidden[:, -1:, :])
                prediction = decode_note(
                    {
                        name: tensor[0, -1, :].tolist()
                        for name, tensor in head_logits.items()
                    }
                )
                symbol.notation = prediction.notation
                symbol.structured_choices = prediction.choices
            symbols.append(symbol)

            out_lift = torch.cat((out_lift, lift_sample), dim=-1)
            out_pitch = torch.cat((out_pitch, pitch_sample), dim=-1)
            out_rhythm = torch.cat((out_rhythm, rhythm_sample), dim=-1)
            out_articulations = torch.cat((out_articulations, articulation_sample), dim=-1)
            out_slurs = torch.cat((out_slurs, slur_sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

        self.net.train(was_training)
        return symbols

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        slurs: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        sampling_prob: float = 1.0,
        coherence_curve: torch.Tensor | None = None,
        coherence_curve_len: torch.Tensor | None = None,
        coherence_present: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        profile_context_emb = self._pop_profile_context_emb(kwargs)
        liftsi = lifts[:, :-1]
        liftso = lifts[:, 1:]
        articulationsi = articulations[:, :-1]
        articulationso = articulations[:, 1:]
        slursi = slurs[:, :-1]
        slurso = slurs[:, 1:]
        pitchsi = pitchs[:, :-1]
        pitchso = pitchs[:, 1:]
        rhythmsi = rhythms[:, :-1]
        rhythmso = rhythms[:, 1:]
        positionsi = positions[:, :-1]
        positionso = positions[:, 1:]

        if mask.shape[1] == rhythms.shape[1]:
            mask = mask[:, :-1]
        mask = mask.bool()

        # Scheduled Sampling: Two-pass approach
        if self.training and sampling_prob < 1.0:
            with torch.no_grad():
                self.net.eval()
                # First pass to get predictions
                r_logits, p_logits, l_logits, pos_logits, a_logits, s_logits, _, _, _ = self.net(
                    rhythms=rhythmsi,
                    pitchs=pitchsi,
                    lifts=liftsi,
                    articulations=articulationsi,
                    slurs=slursi,
                    mask=mask,
                    cache=None,
                    return_center_of_attention=False,
                    profile_context_emb=profile_context_emb,
                    **kwargs,
                )
                self.net.train()

                # Greedy sampling (excluding BOS at index 0)
                # logits[:, t] predicts tokens[:, t+1] (which is input[:, t+1])
                # So logits[:, :-1] corresponds to inputs[:, 1:]
                r_sample = r_logits[:, :-1].argmax(dim=-1)
                p_sample = p_logits[:, :-1].argmax(dim=-1)
                l_sample = l_logits[:, :-1].argmax(dim=-1)
                a_sample = a_logits[:, :-1].argmax(dim=-1)
                s_sample = s_logits[:, :-1].argmax(dim=-1)
                pos_sample = pos_logits[:, :-1].argmax(dim=-1)

                # Determine which indices to replace
                mix_mask = (
                    torch.rand(r_sample.shape, device=rhythms.device) > sampling_prob
                ).long()

                # Mix inputs
                rhythmsi = rhythmsi.clone()
                rhythmsi[:, 1:] = (1 - mix_mask) * rhythmsi[:, 1:] + mix_mask * r_sample

                pitchsi = pitchsi.clone()
                pitchsi[:, 1:] = (1 - mix_mask) * pitchsi[:, 1:] + mix_mask * p_sample

                liftsi = liftsi.clone()
                liftsi[:, 1:] = (1 - mix_mask) * liftsi[:, 1:] + mix_mask * l_sample

                articulationsi = articulationsi.clone()
                articulationsi[:, 1:] = (1 - mix_mask) * articulationsi[:, 1:] + mix_mask * a_sample

                slursi = slursi.clone()
                slursi[:, 1:] = (1 - mix_mask) * slursi[:, 1:] + mix_mask * s_sample

                positionsi = positionsi.clone()
                positionsi[:, 1:] = (1 - mix_mask) * positionsi[:, 1:] + mix_mask * pos_sample

        # Second pass (or standard pass) with (possibly mixed) inputs
        rhythmsp, pitchsp, liftsp, positionsp, articulationsp, slursp, x, _attention, _cache = (
            self.net(
                rhythms=rhythmsi,
                pitchs=pitchsi,
                lifts=liftsi,
                articulations=articulationsi,
                slurs=slursi,
                mask=mask,
                cache=None,
                return_center_of_attention=False,
                profile_context_emb=profile_context_emb,
                **kwargs,
            )
        )  # this calls ScoreTransformerWrapper.forward

        # From the TR OMR paper equation 2, we use however different values for alpha and beta
        alpha = 1
        beta = 1
        loss_consist = beta * self.calConsistencyLoss(
            rhythmsp, pitchsp, liftsp, positionsp, articulationsp, slursp, mask
        )
        # Must run on the still-raw (unmasked) rhythmso - _mask_padding_tokens below
        # replaces padding with ignore_index, which would index out of bounds into
        # rhythm_duration/rhythm_is_barline. 0.0 weight (the default) is exactly the
        # existing loss, bit-for-bit - see calDurationAdherenceLoss's own docstring.
        duration_adherence_weight = getattr(self.config, "duration_adherence_weight", 0.0)
        loss_duration_adherence = (
            duration_adherence_weight
            * self.calDurationAdherenceLoss(rhythmsp, rhythmso, mask)
            if duration_adherence_weight
            else torch.zeros((), device=rhythmsp.device)
        )
        # Same "must run on the still-raw rhythmso" reasoning as duration_adherence
        # above - this loss also indexes rhythm_is_barline by rhythmso_raw.
        cross_staff_coherence_weight = getattr(self.config, "cross_staff_coherence_weight", 0.0)
        loss_cross_staff_coherence = (
            cross_staff_coherence_weight
            * self.calCrossStaffCoherenceLoss(
                rhythmsp, rhythmso, mask, coherence_curve, coherence_curve_len, coherence_present
            )
            if cross_staff_coherence_weight and coherence_curve is not None
            else torch.zeros((), device=rhythmsp.device)
        )
        rhythmso = self._mask_padding_tokens(rhythmso, mask)
        pitchso = self._mask_padding_tokens(pitchso, mask)
        liftso = self._mask_padding_tokens(liftso, mask)
        articulationso = self._mask_padding_tokens(articulationso, mask)
        slurso = self._mask_padding_tokens(slurso, mask)
        positionso = self._mask_padding_tokens(positionso, mask)
        loss_rhythm = alpha * self.cross_entropy(rhythmsp, rhythmso, label_smoothing=0.1)
        loss_pitch = alpha * self.cross_entropy(pitchsp, pitchso)
        loss_lift = alpha * self.cross_entropy(liftsp, liftso)
        loss_articulations = alpha * self.cross_entropy(articulationsp, articulationso)
        loss_slurs = alpha * self.cross_entropy(slursp, slurso)
        loss_position = alpha * self.cross_entropy(positionsp, positionso)
        loss = (
            loss_rhythm
            + loss_pitch
            + loss_lift
            + loss_articulations
            + loss_slurs
            + loss_position
            + loss_consist
            + loss_duration_adherence
            + loss_cross_staff_coherence
        )

        return {
            "loss_rhythm": loss_rhythm,
            "loss_pitch": loss_pitch,
            "loss_lift": loss_lift,
            "loss_consist": loss_consist,
            "loss_duration_adherence": loss_duration_adherence,
            "loss_cross_staff_coherence": loss_cross_staff_coherence,
            "loss_position": loss_position,
            "loss_articulations": loss_articulations,
            "loss_slurs": loss_slurs,
            "loss": loss,
            "logits": (rhythmsp, pitchsp, liftsp, positionsp, articulationsp, slursp),
            # Additive: the shared hidden state, and the structured heads' logits when
            # they exist. The structured loss is deliberately not folded into "loss" -
            # the frozen-core experiment needs the existing objective to stay exactly
            # what it was.
            "hidden": x,
            "structured_logits": (
                self.structured_heads(x) if self.structured_heads is not None else None
            ),
        }

    def calConsistencyLoss(
        self,
        rhythmsp: torch.Tensor,
        pitchsp: torch.Tensor,
        liftsp: torch.Tensor,
        positionsp: torch.Tensor,
        articulationsp: torch.Tensor,
        slursp: torch.Tensor,
        mask: torch.Tensor,
        gamma: int = 10,
    ) -> torch.Tensor:
        positionsp_soft = torch.softmax(positionsp, dim=2)
        positionsp_note = torch.sum(positionsp_soft[:, :, 1:], dim=2) * mask

        rhythmsp_soft = torch.softmax(rhythmsp, dim=2)
        rhythmsp_note = torch.sum(rhythmsp_soft * self.note_mask, dim=2) * mask

        pitchsp_soft = torch.softmax(pitchsp, dim=2)
        pitchsp_note = torch.sum(pitchsp_soft[:, :, 1:], dim=2) * mask

        liftsp_soft = torch.softmax(liftsp, dim=2)
        liftsp_note = torch.sum(liftsp_soft[:, :, 1:], dim=2) * mask

        articulationsp_soft = torch.softmax(articulationsp, dim=2)
        articulationsp_note = torch.sum(articulationsp_soft[:, :, 1:], dim=2) * mask

        slursp_soft = torch.softmax(slursp, dim=2)
        slursp_note = torch.sum(slursp_soft[:, :, 1:], dim=2) * mask

        loss = (
            gamma
            * (
                F.l1_loss(rhythmsp_note, positionsp_note, reduction="none")
                + F.l1_loss(positionsp_note, liftsp_note, reduction="none")
                + F.l1_loss(positionsp_note, pitchsp_note, reduction="none")
                + F.l1_loss(positionsp_note, articulationsp_note, reduction="none")
                + F.l1_loss(positionsp_note, slursp_note, reduction="none")
            )
            / 5.0
        )

        # Apply the mask to the loss and average over the non-masked elements
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def calDurationAdherenceLoss(
        self, rhythmsp: torch.Tensor, rhythmso_raw: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """DECODER_RHYTHM_ACCURACY_DESIGN.md §7.3's refinement: a differentiable
        analogue of Stage A's `check_measure_durations`/`_cumulative_barline_positions`
        (`homr/cross_staff_consistency.py`), penalizing the rhythm head's own predicted
        *expected* cumulative duration for diverging from the ground-truth cumulative
        duration at each true barline position - training-time pressure toward getting
        measure lengths right, not just post-hoc detection of when they're wrong.

        Deliberately compares *cumulative* position at each barline checkpoint, not
        each individual measure's own length in isolation - the same choice
        `_cumulative_barline_positions` itself makes (see its own docstring): if every
        checkpoint's running total matches, every measure between checkpoints must
        also be right, and this needs no per-measure segmentation logic to compute.

        `rhythmso_raw` must be the *unmasked* ground-truth rhythm labels (i.e. `rhythms
        [:, 1:]` before `_mask_padding_tokens` replaces padding with `ignore_index`) -
        the padding-index value would otherwise index out of `rhythm_duration`/
        `rhythm_is_barline`'s bounds. `mask` (the same padding mask every other loss
        here already uses) is applied separately, exactly as `calConsistencyLoss` does.
        """
        not_continuation = self._not_chord_continuation(rhythmso_raw)
        true_duration = self.rhythm_duration[rhythmso_raw] * mask * not_continuation
        probs = torch.softmax(rhythmsp, dim=2)
        predicted_duration = (probs * self.rhythm_duration).sum(dim=2) * mask * not_continuation

        true_cumulative = torch.cumsum(true_duration, dim=1)
        predicted_cumulative = torch.cumsum(predicted_duration, dim=1)

        is_barline = self.rhythm_is_barline[rhythmso_raw] * mask
        diff = torch.abs(predicted_cumulative - true_cumulative) * is_barline
        denom = is_barline.sum().clamp(min=1)
        return diff.sum() / denom

    def _not_chord_continuation(self, rhythmso_raw: torch.Tensor) -> torch.Tensor:
        """`1.0` everywhere except a position immediately following a "chord" marker
        token - see the long comment on `rhythm_duration`'s construction above for why
        that position must not contribute its own duration a second time. Computed
        from `rhythmso_raw` (ground truth), the same source `is_barline` is keyed to,
        so both the true and predicted duration sums are masked at identical positions."""
        is_chord_marker = self.rhythm_is_chord_marker[rhythmso_raw]
        is_continuation = torch.zeros_like(is_chord_marker)
        is_continuation[:, 1:] = is_chord_marker[:, :-1]
        return 1.0 - is_continuation

    def calCrossStaffCoherenceLoss(
        self,
        rhythmsp: torch.Tensor,
        rhythmso_raw: torch.Tensor,
        mask: torch.Tensor,
        coherence_curve: torch.Tensor,
        coherence_curve_len: torch.Tensor,
        coherence_present: torch.Tensor,
    ) -> torch.Tensor:
        """DECODER_RHYTHM_ACCURACY_DESIGN.md §7.3's loss brainstorm item 2: like
        `calDurationAdherenceLoss`, but the target at each barline is this staff's
        *system's* ground truth (the median across every sibling part,
        `training.omr_datasets.cross_staff_coherence.system_measure_curve`) rather than
        this staff's own label - a real, cheaper alternative to §4 Stage C's learned
        adapter worth trying first: no architecture change, no inference-time coupling,
        just a training signal that teaches the model to prefer duration decisions
        that stay coherent with what a system's siblings actually contain.

        `coherence_curve` is `(batch, MAX_COHERENCE_MEASURES)`, this sample's own
        system's whole-note cumulative-duration curve, padded/truncated to that fixed
        width by the data loader; `coherence_curve_len` `(batch,)` is how many of those
        entries are real (the rest is padding, never indexed past this); `coherence_
        present` `(batch,)` is 0 for a sample whose system curve could not be resolved
        at all (an unresolvable score, an unaligned page/system - the same "unknown is
        never a guess" discipline `time_signature_for_sample` uses) - such a sample
        contributes nothing here, not a fabricated zero target.

        `rhythmso_raw` must be the same still-*unmasked* ground-truth rhythm labels
        `calDurationAdherenceLoss` requires, for the same reason (indexing
        `rhythm_is_barline`).
        """
        not_continuation = self._not_chord_continuation(rhythmso_raw)
        probs = torch.softmax(rhythmsp, dim=2)
        predicted_duration = (probs * self.rhythm_duration).sum(dim=2) * mask * not_continuation
        predicted_cumulative = torch.cumsum(predicted_duration, dim=1)

        is_barline = self.rhythm_is_barline[rhythmso_raw] * mask
        # How many of *this sample's own* barlines have been crossed by each position -
        # the 1st barline crossed should match the system curve's 0th measure entry.
        barline_count = torch.cumsum(is_barline, dim=1)
        curve_index = (barline_count - 1).clamp(min=0).long()
        max_index = (coherence_curve_len - 1).clamp(min=0).view(-1, 1)
        curve_index = torch.minimum(curve_index, max_index.expand_as(curve_index))
        target = torch.gather(coherence_curve, 1, curve_index)

        in_range = (barline_count <= coherence_curve_len.view(-1, 1).clamp(min=0)).float()
        valid = is_barline * in_range * coherence_present.view(-1, 1)
        diff = torch.abs(predicted_cumulative - target) * valid
        denom = valid.sum().clamp(min=1)
        return diff.sum() / denom

    def cross_entropy(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        label_smoothing: float = 0.0,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits.transpose(1, 2),
            target,
            reduction="mean",
            weight=weights,
            ignore_index=self.ignore_index,
            label_smoothing=label_smoothing,
        )

    def _mask_padding_tokens(self, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return target.masked_fill(~mask, self.ignore_index)


def init_cache(
    cache_len: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[str], list[str], dict[str, dict[int, str]], int]:
    cache = []
    input_names = []
    output_names = []
    dynamic = {}
    for i in range(32):
        cache.append(torch.zeros((1, 8, cache_len, 64), dtype=torch.float32).to(device))
        input_names.append(f"cache_in{i}")
        output_names.append(f"cache_out{i}")
        dynamic[f"cache_in{i}"] = {2: "seq_len"}
    return cache, input_names, output_names, dynamic, cache_len


def get_decoder(config: Config) -> ScoreDecoder:
    return ScoreDecoder(
        get_score_wrapper(config),
        config=config,
    )


def get_score_wrapper(config: Config, attn_flash: bool = True) -> ScoreTransformerWrapper:
    return ScoreTransformerWrapper(
        config=config,
        attn_layers=CustomDecoder(
            dim=config.decoder_dim,
            depth=config.decoder_depth,
            heads=config.decoder_heads,
            attn_flash=attn_flash,
            **config.decoder_args.to_dict(),
        ),
    )


def detokenize(tokens: torch.Tensor, vocab: Any) -> list[str]:
    toks = [vocab[tok.item()] for tok in tokens]
    toks = [t for t in toks if t not in ("[BOS]", "[EOS]", "[PAD]")]
    return toks
