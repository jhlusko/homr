from typing import Any

import numpy as np
import onnxruntime as ort

from homr.onnx_providers import gpu_providers
from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray


class ScoreDecoder:
    def __init__(
        self,
        transformer: ort.InferenceSession,
        fp16: bool,
        use_gpu: bool,
        config: Config,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.config = config
        self.net = transformer
        self.io_binding = self.net.io_binding()
        self.max_seq_len = config.max_seq_len
        self.eos_token = config.eos_token

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}
        self.inv_articulation_vocab = {v: k for k, v in config.articulation_vocab.items()}
        self.inv_slur_vocab = {v: k for k, v in config.slur_vocab.items()}
        self.inv_position_vocab = {v: k for k, v in config.position_vocab.items()}

        self.fp16 = fp16
        self.use_gpu = use_gpu
        self.device_id = 0
        self.output_names = [
            "out_rhythms",
            "out_pitchs",
            "out_lifts",
            "out_positions",
            "out_articulations",
            "out_slurs",
            "attention",
            "hidden",
        ]
        # §4/§7.4 Stage C: every call binds this ONNX input (added by
        # training/onnx/convert.py's DecoderWrapper) - a zero vector reproduces the
        # exact pre-Stage-C graph (ScoreTransformerWrapper's own zero-bias-is-a-no-op
        # guarantee), which is what every method below still does unless a caller
        # explicitly passes a real one via `staff_context_emb=`.
        self._zero_staff_context_emb = np.zeros(
            (1, config.decoder_dim), dtype=np.float16 if fp16 else np.float32
        )

    def generate(
        self,
        start_tokens: NDArray,
        nonote_tokens: NDArray,
        **kwargs: Any,
    ) -> list[EncodedSymbol]:
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        out_rhythm = start_tokens
        out_pitch = nonote_tokens
        out_lift = nonote_tokens
        out_articulations = nonote_tokens
        out_slurs = nonote_tokens
        cache, kv_input_names, kv_output_names = self.init_cache()
        output_names = self.output_names + kv_output_names
        context = kwargs["context"]
        context_reduced = kwargs["context"][:, :1]
        staff_context_emb = kwargs.get("staff_context_emb", self._zero_staff_context_emb)

        symbols: list[EncodedSymbol] = []

        for step in range(self.max_seq_len):
            x_lift = out_lift[:, -1:]  # for all: shape=(1,1)
            x_pitch = out_pitch[:, -1:]
            x_rhythm = out_rhythm[:, -1:]
            x_articulations = out_articulations[:, -1:]
            x_slurs = out_slurs[:, -1:]

            # after the first step we don't pass the full context into the decoder
            # x_transformers uses [:, :0] to split the context
            # which caused a Reshape error when loading the onnx model
            context = context if step == 0 else context_reduced

            # Bind Inputs
            self.io_binding.bind_cpu_input("rhythms", x_rhythm)
            self.io_binding.bind_cpu_input("pitchs", x_pitch)
            self.io_binding.bind_cpu_input("lifts", x_lift)
            self.io_binding.bind_cpu_input("articulations", x_articulations)
            self.io_binding.bind_cpu_input("slurs", x_slurs)
            self.io_binding.bind_cpu_input("context", context)
            self.io_binding.bind_cpu_input("cache_len", np.array([step], dtype=np.int64))
            self.io_binding.bind_cpu_input("staff_context_emb", staff_context_emb)
            for name, cache_val in zip(kv_input_names, cache, strict=True):
                self.io_binding.bind_ortvalue_input(name, cache_val)

            # Bind Outputs
            for name in output_names:
                self.io_binding.bind_output(name, "cuda" if self.use_gpu else "cpu", self.device_id)

            # Run inference
            self.net.run_with_iobinding(iobinding=self.io_binding)

            # Get outputs
            outputs = self.io_binding.get_outputs()
            cache = outputs[8:]

            # Greedy decoding: pick the highest logit directly for each output
            rhythmsp = outputs[0].numpy()
            pitchsp = outputs[1].numpy()
            liftsp = outputs[2].numpy()
            positionsp = outputs[3].numpy()
            articulationsp = outputs[4].numpy()
            slursp = outputs[5].numpy()
            attention = outputs[6].numpy()

            rhythm_sample = np.array([[rhythmsp[:, -1, :].argmax()]])
            pitch_sample = np.array([[pitchsp[:, -1, :].argmax()]])
            lift_sample = np.array([[liftsp[:, -1, :].argmax()]])
            articulation_sample = np.array([[articulationsp[:, -1, :].argmax()]])
            slur_sample = np.array([[slursp[:, -1, :].argmax()]])
            position_sample = np.array([[positionsp[:, -1, :].argmax()]])

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
                coordinates=attention,
            )
            symbols.append(symbol)

            out_lift = np.concatenate((out_lift, lift_sample), axis=-1)
            out_pitch = np.concatenate((out_pitch, pitch_sample), axis=-1)
            out_rhythm = np.concatenate((out_rhythm, rhythm_sample), axis=-1)
            out_articulations = np.concatenate((out_articulations, articulation_sample), axis=-1)
            out_slurs = np.concatenate((out_slurs, slur_sample), axis=-1)

        return symbols

    def generate_with_rhythm_margins(
        self,
        start_tokens: NDArray,
        nonote_tokens: NDArray,
        **kwargs: Any,
    ) -> tuple[list[EncodedSymbol], list[tuple[int, float]], NDArray]:
        """Identical greedy decode to `generate()`, plus bookkeeping `generate()` itself
        never needed: at every step, the rhythm head's second-best token id and the
        logit margin between the chosen (top-1) and runner-up (top-2) rhythm token.

        This is the raw material Phase 1's cross-staff-consistency reranking
        (`DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.2,
        `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §5) needs to find which rhythm decisions
        were a narrow, close call - a small margin at step `i` means the model's second
        choice at that step was nearly as likely as what it actually picked, exactly the
        "one mis-decoded note, could easily have gone the other way" case reranking
        targets. A large margin means the model was confident, and forking there is very
        unlikely to help while still costing a full re-decode - `rerank_staff_candidates`
        (`homr/cross_staff_rerank.py`) uses this to pick which steps are worth the cost
        of actually branching on via `rhythm_alternative`, rather than forking at every
        step.

        Returns `(symbols, margins, hidden_states)` where `margins[i]` is
        `(alt_rhythm_token_id, margin)` for the step that produced `symbols[i]` -
        aligned 1:1, not a separate indexing scheme. Changes no decoding decision from
        `generate()` - the chosen token at every step is still the same top-1 argmax;
        this only records what the alternative would have been and how close it was.

        `hidden_states` is `(steps, decoder_dim)`, one row per step in `symbols` (the
        BOS step is not included) - §4/§7.4 Stage C's own first-pass input, the same
        `"hidden"`/`x` key `ScoreDecoder.forward`'s training-time counterpart already
        exposes for the same purpose, collected here since inference has no single
        forward call over the whole sequence to read it from afterward.
        """
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        out_rhythm = start_tokens
        out_pitch = nonote_tokens
        out_lift = nonote_tokens
        out_articulations = nonote_tokens
        out_slurs = nonote_tokens
        cache, kv_input_names, kv_output_names = self.init_cache()
        output_names = self.output_names + kv_output_names
        context = kwargs["context"]
        context_reduced = kwargs["context"][:, :1]
        staff_context_emb = kwargs.get("staff_context_emb", self._zero_staff_context_emb)

        symbols: list[EncodedSymbol] = []
        margins: list[tuple[int, float]] = []
        hidden_states: list[NDArray] = []

        for step in range(self.max_seq_len):
            x_lift = out_lift[:, -1:]
            x_pitch = out_pitch[:, -1:]
            x_rhythm = out_rhythm[:, -1:]
            x_articulations = out_articulations[:, -1:]
            x_slurs = out_slurs[:, -1:]

            context_input = context if step == 0 else context_reduced

            self.io_binding.bind_cpu_input("rhythms", x_rhythm)
            self.io_binding.bind_cpu_input("pitchs", x_pitch)
            self.io_binding.bind_cpu_input("lifts", x_lift)
            self.io_binding.bind_cpu_input("articulations", x_articulations)
            self.io_binding.bind_cpu_input("slurs", x_slurs)
            self.io_binding.bind_cpu_input("context", context_input)
            self.io_binding.bind_cpu_input("cache_len", np.array([step], dtype=np.int64))
            self.io_binding.bind_cpu_input("staff_context_emb", staff_context_emb)
            for name, cache_val in zip(kv_input_names, cache, strict=True):
                self.io_binding.bind_ortvalue_input(name, cache_val)

            for name in output_names:
                self.io_binding.bind_output(name, "cuda" if self.use_gpu else "cpu", self.device_id)

            self.net.run_with_iobinding(iobinding=self.io_binding)

            outputs = self.io_binding.get_outputs()
            cache = outputs[8:]

            rhythmsp = outputs[0].numpy()
            pitchsp = outputs[1].numpy()
            liftsp = outputs[2].numpy()
            positionsp = outputs[3].numpy()
            articulationsp = outputs[4].numpy()
            slursp = outputs[5].numpy()
            attention = outputs[6].numpy()
            hidden = outputs[7].numpy()

            rhythm_logits = rhythmsp[:, -1, :].reshape(-1)
            top2_idx = np.argsort(rhythm_logits)[-2:]
            alt_token_id = int(top2_idx[0])
            top_token_id = int(top2_idx[1])
            margin = float(rhythm_logits[top_token_id] - rhythm_logits[alt_token_id])

            rhythm_sample = np.array([[top_token_id]])
            pitch_sample = np.array([[pitchsp[:, -1, :].argmax()]])
            lift_sample = np.array([[liftsp[:, -1, :].argmax()]])
            articulation_sample = np.array([[articulationsp[:, -1, :].argmax()]])
            slur_sample = np.array([[slursp[:, -1, :].argmax()]])
            position_sample = np.array([[positionsp[:, -1, :].argmax()]])

            if rhythm_sample[0][0] == self.eos_token:
                break

            symbols.append(
                EncodedSymbol(
                    rhythm=detokenize(rhythm_sample, self.inv_rhythm_vocab)[0],
                    pitch=detokenize(pitch_sample, self.inv_pitch_vocab)[0],
                    lift=detokenize(lift_sample, self.inv_lift_vocab)[0],
                    articulation=detokenize(articulation_sample, self.inv_articulation_vocab)[0],
                    slur=detokenize(slur_sample, self.inv_slur_vocab)[0],
                    position=detokenize(position_sample, self.inv_position_vocab)[0],
                    coordinates=attention,
                )
            )
            margins.append((alt_token_id, margin))
            hidden_states.append(hidden[:, -1, :])

            out_lift = np.concatenate((out_lift, lift_sample), axis=-1)
            out_pitch = np.concatenate((out_pitch, pitch_sample), axis=-1)
            out_rhythm = np.concatenate((out_rhythm, rhythm_sample), axis=-1)
            out_articulations = np.concatenate((out_articulations, articulation_sample), axis=-1)
            out_slurs = np.concatenate((out_slurs, slur_sample), axis=-1)

        stacked_hidden = (
            np.concatenate(hidden_states, axis=0)
            if hidden_states
            else np.zeros((0, self.config.decoder_dim), dtype=np.float32)
        )
        return symbols, margins, stacked_hidden

    def rhythm_alternative(
        self,
        symbols: list[EncodedSymbol],
        fork_step: int,
        alt_rhythm_token_id: int,
        **kwargs: Any,
    ) -> list[EncodedSymbol]:
        """Builds the full alternate decode produced by swapping the rhythm token at
        `fork_step` (0-indexed into `symbols`, `generate_with_rhythm_margins`'s output)
        for `alt_rhythm_token_id`, then regenerating everything after it - on top of
        `generate_from_prefix`, already validated against the live model for exactly
        this teacher-forced-prefix-then-continue mechanism (see its own docstring).
        Every other field at `fork_step` (pitch/lift/articulation/slur/position) is left
        as originally decoded; only the rhythm token is corrected, matching how the
        model actually consumes these fields (each is its own autoregressive input
        stream, only tied together through the shared forward pass, not through one
        field's history influencing another's within the same step).
        """
        bos = EncodedSymbol(
            rhythm=self.inv_rhythm_vocab[1],
            pitch=self.inv_pitch_vocab[0],
            lift=self.inv_lift_vocab[0],
            articulation=self.inv_articulation_vocab[0],
            slur=self.inv_slur_vocab[0],
        )
        corrected = EncodedSymbol(
            rhythm=self.inv_rhythm_vocab[alt_rhythm_token_id],
            pitch=symbols[fork_step].pitch,
            lift=symbols[fork_step].lift,
            articulation=symbols[fork_step].articulation,
            slur=symbols[fork_step].slur,
            position=symbols[fork_step].position,
        )
        prefix = [bos, *symbols[:fork_step], corrected]
        continuation = self.generate_from_prefix(prefix, **kwargs)
        return [*symbols[:fork_step], corrected, *continuation]

    def generate_from_prefix(
        self, prefix: list[EncodedSymbol], **kwargs: Any
    ) -> list[EncodedSymbol]:
        """Regenerate a staff's decode from a corrected point onward - tier 2 of the
        Stage B refinement (design; ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md §4).

        A bare top-k token swap only fixes the one token it swaps; everything decoded
        after it was chosen under the *original*, wrong context and is not revisited -
        fine for an isolated rhythm error, wrong for something like a key signature that
        could have shaped every accidental spelling that followed it. `generate()`
        cannot do better because it always starts from a single BOS seed with an empty
        cache. This method teacher-forces `prefix` through the model one token at a
        time - the same causal decoder, the same KV cache mechanics, just fed the
        caller's known-correct values instead of letting each step sample its own - so
        the cache ends up reflecting the corrected sequence exactly, then falls through
        to ordinary greedy decoding once the prefix is exhausted. No retraining: this is
        the frozen decoder `generate()` already uses, run the way it is already able to
        run.

        `prefix` must start with a BOS symbol (`EncodedSymbol("BOS")`) followed by
        everything up to and including the corrected token, in the same shape
        `generate()`'s own `start_tokens` seed takes (BOS, id 1, before the loop starts) -
        this function does not invent a separate seeding convention. Returns only the
        newly generated continuation; the caller already has `prefix` and splices the two
        together.

        Validated against the real ONNX model on two counts, not just unit-tested in
        isolation (this module has no mocked test coverage at all - the IO-binding/cache
        contract is exactly the kind of thing a mock would risk getting subtly wrong):
        replaying a staff's own first K decoded symbols as a forced prefix and
        continuing reproduces the rest of that staff's decode bit-for-bit identical to
        an unforced `generate()` call - confirms the cache/step/position mechanics are
        correct, not merely plausible. Separately, forcing a genuinely different key
        signature (`keySignature_-1` -> `keySignature_4`, mid-staff) left every
        downstream pitch and accidental (`lift`) prediction unchanged on the one staff
        tested - a real, measured finding that this model's pitch/accidental decisions
        are apparently driven by the visual encoder context more than by the key
        signature token in the autoregressive history, not a defect in this method. It
        means tier 2 is mechanically sound and available, but its practical value for
        the key/time-signature repair case specifically is not yet established and
        needs testing across more staves before being relied on for that use case.
        """
        if not prefix:
            raise ValueError("prefix must contain at least a BOS token")

        context = kwargs["context"]
        context_reduced = kwargs["context"][:, :1]
        cache, kv_input_names, kv_output_names = self.init_cache()
        output_names = self.output_names + kv_output_names

        out_rhythm = np.array([[self._token_id(self.config.rhythm_vocab, prefix[0].rhythm)]])
        out_pitch = np.array([[self._token_id(self.config.pitch_vocab, prefix[0].pitch)]])
        out_lift = np.array([[self._token_id(self.config.lift_vocab, prefix[0].lift)]])
        out_articulations = np.array(
            [[self._token_id(self.config.articulation_vocab, prefix[0].articulation)]]
        )
        out_slurs = np.array([[self._token_id(self.config.slur_vocab, prefix[0].slur)]])

        symbols: list[EncodedSymbol] = []

        for step in range(self.max_seq_len):
            x_lift = out_lift[:, -1:]
            x_pitch = out_pitch[:, -1:]
            x_rhythm = out_rhythm[:, -1:]
            x_articulations = out_articulations[:, -1:]
            x_slurs = out_slurs[:, -1:]

            context_input = context if step == 0 else context_reduced

            self.io_binding.bind_cpu_input("rhythms", x_rhythm)
            self.io_binding.bind_cpu_input("pitchs", x_pitch)
            self.io_binding.bind_cpu_input("lifts", x_lift)
            self.io_binding.bind_cpu_input("articulations", x_articulations)
            self.io_binding.bind_cpu_input("slurs", x_slurs)
            self.io_binding.bind_cpu_input("context", context_input)
            self.io_binding.bind_cpu_input("cache_len", np.array([step], dtype=np.int64))
            self.io_binding.bind_cpu_input("staff_context_emb", self._zero_staff_context_emb)
            for name, cache_val in zip(kv_input_names, cache, strict=True):
                self.io_binding.bind_ortvalue_input(name, cache_val)

            for name in output_names:
                self.io_binding.bind_output(name, "cuda" if self.use_gpu else "cpu", self.device_id)

            self.net.run_with_iobinding(iobinding=self.io_binding)

            outputs = self.io_binding.get_outputs()
            cache = outputs[8:]

            rhythmsp = outputs[0].numpy()
            pitchsp = outputs[1].numpy()
            liftsp = outputs[2].numpy()
            positionsp = outputs[3].numpy()
            articulationsp = outputs[4].numpy()
            slursp = outputs[5].numpy()
            attention = outputs[6].numpy()

            # Step 0 fed prefix[0]; the next still-known token, if any, is prefix[step+1].
            next_known = prefix[step + 1] if step + 1 < len(prefix) else None

            if next_known is not None:
                # Still inside the forced prefix: ignore the model's own prediction and
                # feed the caller's known-correct value forward instead - teacher
                # forcing, so the cache ends up reflecting the corrected sequence rather
                # than whatever the model would have guessed.
                rhythm_sample = np.array(
                    [[self._token_id(self.config.rhythm_vocab, next_known.rhythm)]]
                )
                pitch_sample = np.array(
                    [[self._token_id(self.config.pitch_vocab, next_known.pitch)]]
                )
                lift_sample = np.array([[self._token_id(self.config.lift_vocab, next_known.lift)]])
                articulation_sample = np.array(
                    [[self._token_id(self.config.articulation_vocab, next_known.articulation)]]
                )
                slur_sample = np.array([[self._token_id(self.config.slur_vocab, next_known.slur)]])
            else:
                # Prefix exhausted: ordinary greedy decoding, identical to generate().
                rhythm_sample = np.array([[rhythmsp[:, -1, :].argmax()]])
                pitch_sample = np.array([[pitchsp[:, -1, :].argmax()]])
                lift_sample = np.array([[liftsp[:, -1, :].argmax()]])
                articulation_sample = np.array([[articulationsp[:, -1, :].argmax()]])
                slur_sample = np.array([[slursp[:, -1, :].argmax()]])
                position_sample = np.array([[positionsp[:, -1, :].argmax()]])

                if rhythm_sample[0][0] == self.eos_token:
                    break

                symbols.append(
                    EncodedSymbol(
                        rhythm=detokenize(rhythm_sample, self.inv_rhythm_vocab)[0],
                        pitch=detokenize(pitch_sample, self.inv_pitch_vocab)[0],
                        lift=detokenize(lift_sample, self.inv_lift_vocab)[0],
                        articulation=detokenize(
                            articulation_sample, self.inv_articulation_vocab
                        )[0],
                        slur=detokenize(slur_sample, self.inv_slur_vocab)[0],
                        position=detokenize(position_sample, self.inv_position_vocab)[0],
                        coordinates=attention,
                    )
                )

            out_lift = np.concatenate((out_lift, lift_sample), axis=-1)
            out_pitch = np.concatenate((out_pitch, pitch_sample), axis=-1)
            out_rhythm = np.concatenate((out_rhythm, rhythm_sample), axis=-1)
            out_articulations = np.concatenate((out_articulations, articulation_sample), axis=-1)
            out_slurs = np.concatenate((out_slurs, slur_sample), axis=-1)

        return symbols

    @staticmethod
    def _token_id(vocab: dict[str, int], value: str) -> int:
        if value not in vocab:
            raise ValueError(f"{value!r} is not in this decoder's vocabulary")
        return vocab[value]

    def init_cache(self, cache_len: int = 0) -> tuple[list[NDArray], list[str], list[str]]:
        cache = []
        input_names = []
        output_names = []
        heads = self.config.decoder_heads
        head_dim = self.config.decoder_dim // heads
        for i in range(self.config.decoder_depth * 4):
            if self.fp16:  # the cache needs to be fp16 as well
                cache.append(
                    ort.OrtValue.ortvalue_from_numpy(
                        np.zeros((1, heads, cache_len, head_dim), dtype=np.float16),
                        "cuda" if self.use_gpu else "cpu",
                        self.device_id,
                    )
                )
            else:
                cache.append(
                    ort.OrtValue.ortvalue_from_numpy(
                        np.zeros((1, heads, cache_len, head_dim), dtype=np.float32),
                        "cuda" if self.use_gpu else "cpu",
                        self.device_id,
                    )
                )
            input_names.append(f"cache_in{i}")
            output_names.append(f"cache_out{i}")
        return cache, input_names, output_names


def detokenize(tokens: NDArray, vocab: dict[int, str]) -> list[str]:
    toks = [vocab[tok.item()] for tok in tokens]
    toks = [t for t in toks if t not in ("[BOS]", "[EOS]", "[PAD]")]
    return toks


def get_decoder(config: Config) -> ScoreDecoder:
    """
    Returns Tromr's Decoder
    """
    use_gpu = False
    if config.use_gpu_inference:
        try:
            providers, device = gpu_providers()
            onnx_transformer = ort.InferenceSession(
                config.filepaths.decoder_path_fp16, providers=providers
            )
            fp16 = True
            # Sometimes Ort falls automatically back to the CPU EP
            # if so we get an error due to the device selection in init_cache().
            # CoreML binds IO on the CPU (device == "cpu") even when the GPU/ANE runs the
            # compute, so we only flip use_gpu on when CUDA device memory is in play.
            active = onnx_transformer.get_providers()
            if device == "cuda" and "CUDAExecutionProvider" in active:
                use_gpu = True
            elif device == "cuda":
                eprint(
                    "Onnxruntime is not using GPU and therefore falling back to CPU. This is slow."
                )

        except Exception as ex:
            eprint(ex)
            eprint("Going on without GPU support")
            onnx_transformer = ort.InferenceSession(config.filepaths.decoder_path_fp16)
            fp16 = True

    else:
        onnx_transformer = ort.InferenceSession(config.filepaths.decoder_path)
        fp16 = False

    return ScoreDecoder(onnx_transformer, fp16, use_gpu, config=config)
