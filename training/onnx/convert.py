import os

import torch
from torch.export import Dim

from homr.segmentation.config import segnet_path_onnx, segnet_path_torch
from homr.simple_logging import eprint
from homr.transformer.configs import Config
from training.architecture.transformer.decoder import (
    ScoreTransformerWrapper,
    get_score_wrapper,
    init_cache,
)
from training.architecture.transformer.structured_heads import StructuredNotationHeads


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model: ScoreTransformerWrapper) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        slurs: torch.Tensor,
        context: torch.Tensor,
        cache_len: torch.Tensor,
        staff_context_emb: torch.Tensor,
        *cache: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
    ]:
        # §4/§7.4 Stage C: an all-zero staff_context_emb (every existing caller, until
        # a caller actually runs the two-pass decode) reproduces the pre-Stage-C graph
        # exactly - ScoreTransformerWrapper's own zero-bias-is-a-no-op guarantee
        # (test_staff_context_wiring.py), not a new behavior introduced here.
        (
            out_rhythms,
            out_pitchs,
            out_lifts,
            out_positions,
            out_articulations,
            out_slurs,
            x,
            attention,
            *cache,
        ) = self.model(
            rhythms=rhythms,
            pitchs=pitchs,
            lifts=lifts,
            articulations=articulations,
            slurs=slurs,
            context=context,
            cache_len=cache_len,
            mask=None,
            cache=cache,
            return_center_of_attention=True,
            staff_context_emb=staff_context_emb,
        )
        return (
            out_rhythms,
            out_pitchs,
            out_lifts,
            out_positions,
            out_articulations,
            out_slurs,
            attention,
            x,
            *cache,
        )


def convert_encoder(overwrite: bool, out_dir: str | None = None) -> str | None:
    """
    Converts the encoder to onnx

    `out_dir`, when given, redirects the output away from `config.filepaths.encoder_path`
    - which is also the live cache `download_weights` populates, keyed to the pinned
    architecture's name regardless of whose weights were actually loaded. Exporting a
    non-pinned checkpoint (a fork's own training run, say) without `out_dir` silently
    overwrites that cache in place; see `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` for the
    time that happened. Omitting it keeps the exact behaviour this had before, since that
    is still correct for exporting the pinned checkpoint itself.
    """
    config = Config()

    path_out = (
        os.path.join(out_dir, os.path.basename(config.filepaths.encoder_path))
        if out_dir
        else config.filepaths.encoder_path
    )

    if os.path.exists(path_out) and not overwrite:
        eprint(
            f"Encoder already exists at {path_out}. Use --overwrite to overwrite the existing file."
        )
        return None

    # Get Encoder
    # Local for the same reason as create_segnet below: the encoder imports timm,
    # which the decoder and structured-head exports do not need.
    from training.architecture.transformer.encoder import get_encoder

    model = get_encoder(config)

    # Load weights
    model.load_state_dict(
        torch.load(r"encoder_weights.pt", weights_only=True, map_location=torch.device("cpu")),
        strict=True,
    )

    # Set eval mode
    model.eval()

    # Prepare input tensor
    input_tensor = torch.randn(1, 1, config.max_height, config.max_width).float()

    # Export to onnx
    torch.onnx.export(
        model,
        input_tensor,  # type: ignore
        path_out,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    return path_out


def convert_decoder(overwrite: bool, out_dir: str | None = None) -> str | None:
    """
    Converts the decoder to onnx.

    `out_dir` redirects output away from the live weights cache - see
    `convert_encoder`'s docstring for why that matters.
    """
    config = Config()
    model = get_score_wrapper(config, attn_flash=False)
    model.eval()

    path_out = (
        os.path.join(out_dir, os.path.basename(config.filepaths.decoder_path))
        if out_dir
        else config.filepaths.decoder_path
    )

    if os.path.exists(path_out) and not overwrite:
        eprint(
            f"Decoder already exists at {path_out}. Use --overwrite to overwrite the existing file."
        )
        return None

    model.load_state_dict(
        torch.load(r"decoder_weights.pt", weights_only=True, map_location=torch.device("cpu")),
        strict=True,
    )

    # Using a wrapper model with a custom forward() function
    wrapped_model = DecoderWrapper(model)
    wrapped_model.eval()

    # Create input data
    # Mask is not used since it caused problems with the tensor size
    kv_cache, kv_input_names, kv_output_names, dynamic_axes, cache_length = init_cache(
        0, torch.device("cpu")
    )
    rhythms = torch.randint(0, config.num_rhythm_tokens, (1, 1)).long()
    pitchs = torch.randint(0, config.num_pitch_tokens, (1, 1)).long()
    lifts = torch.randint(0, config.num_lift_tokens, (1, 1)).long()
    articulations = torch.randint(0, config.num_articulation_tokens, (1, 1)).long()
    slurs = torch.randint(0, config.num_slur_tokens, (1, 1)).long()
    cache_len = torch.tensor([cache_length]).long()
    cache = kv_cache
    context = torch.randn((1, 1280, config.encoder_dim)).float()
    # §4/§7.4 Stage C: an all-zero vector here traces the exact same no-op path
    # ScoreTransformerWrapper's own zero-bias guarantee already covers - every
    # existing caller keeps passing zeros until one actually runs the two-pass
    # decode with a real StaffContextTransformer output.
    staff_context_emb = torch.zeros((1, config.decoder_dim)).float()

    dynamic_axes["context"] = {1: "cache_exists"}

    torch.onnx.export(
        wrapped_model,
        (rhythms, pitchs, lifts, articulations, slurs, context, cache_len, staff_context_emb, *cache),
        path_out,
        input_names=[
            "rhythms",
            "pitchs",
            "lifts",
            "articulations",
            "slurs",
            "context",
            "cache_len",
            "staff_context_emb",
            *kv_input_names,
        ],
        output_names=[
            "out_rhythms",
            "out_pitchs",
            "out_lifts",
            "out_positions",
            "out_articulations",
            "out_slurs",
            "attention",
            "hidden",
            *kv_output_names,
        ],
        dynamic_axes=dynamic_axes,
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )
    return path_out


def convert_segnet(overwrite: bool) -> str | None:
    """
    Converts the segnet model to onnx.
    """
    path_out = segnet_path_onnx

    if os.path.exists(path_out) and not overwrite:
        eprint(
            f"Segnet already exists at {path_out}. Use --overwrite to overwrite the existing file."
        )
        return None

    # Imported here rather than at module scope: it pulls in pytorch_lightning and
    # the whole segmentation stack, which the encoder, decoder and structured-head
    # exports have no need of. Keeping it local means those three can be used - and
    # tested - without installing any of it.
    from training.architecture.segmentation.model import create_segnet  # type: ignore

    model = create_segnet()
    model.load_state_dict(torch.load(segnet_path_torch, weights_only=True), strict=True)
    model.eval()

    # Input dimension is 1x3x320x320
    sample_inputs = torch.randn(8, 3, 320, 320)

    torch.onnx.export(
        model,
        sample_inputs,  # type: ignore
        path_out,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # dyamic axes are required for dynamic batch_size
        dynamic_shapes={"image": (Dim("batch_size"), 3, 320, 320)},
        dynamo=True,
        external_data=False,
    )
    return path_out


class StructuredHeadsWrapper(torch.nn.Module):
    """Fixed output order for the heads, which otherwise come back in a dict.

    ONNX graphs have positional outputs, so the head order becomes part of the file's
    contract. Sorting by name makes it reproducible across exports rather than dependent
    on dict insertion order.
    """

    def __init__(self, heads: StructuredNotationHeads, names: list[str]) -> None:
        super().__init__()
        self.heads = heads
        self.names = names

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        logits = self.heads(hidden)
        return tuple(logits[name] for name in self.names)


def convert_structured_heads(
    overwrite: bool, weights: str, out_dir: str | None = None
) -> str | None:
    """Export the structured beam/stem/slur heads as their own ONNX graph.

    Separate from the decoder on purpose. The heads are a non-autoregressive projection
    of the decoder's hidden state - `structuredHeadsAutoregressive: false` in the
    capability manifest - and the decoder graph already exposes `hidden` as an output.
    So they need no change to the decoder export, and a deployment without this file
    behaves exactly as it did before, which is the same rule `configs.py` applies to
    enabling the heads at all.

    `weights` is a heads checkpoint (`heads_clef.pth`), which holds only the head
    tensors; it is meaningless apart from the core it was trained against. `out_dir`
    redirects output away from the live weights cache - see `convert_encoder`'s
    docstring for why that matters.
    """
    config = Config()
    path_out = (
        os.path.join(out_dir, os.path.basename(config.filepaths.structured_heads_path))
        if out_dir
        else config.filepaths.structured_heads_path
    )

    if os.path.exists(path_out) and not overwrite:
        eprint(f"Structured heads already exist at {path_out}. Use --overwrite.")
        return None

    heads = StructuredNotationHeads(
        dim=config.decoder_dim,
        beam_levels=config.structured_beam_levels,
        slur_slots=config.structured_slur_slots,
    )
    state = torch.load(weights, weights_only=True, map_location=torch.device("cpu"))
    # The checkpoint stores the heads under their path in the full model; strip that
    # prefix so the module can be loaded standalone.
    stripped = {key.split("structured_heads.", 1)[-1]: value for key, value in state.items()}
    missing, unexpected = heads.load_state_dict(stripped, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Head weights do not match this config: {len(missing)} missing, "
            f"{len(unexpected)} unexpected. The checkpoint was probably trained with a "
            f"different beam-level or slur-slot count."
        )
    heads.eval()

    names = sorted(heads.head_names())
    hidden = torch.randn((1, 1, config.decoder_dim)).float()

    torch.onnx.export(
        StructuredHeadsWrapper(heads, names),
        (hidden,),
        path_out,
        input_names=["hidden"],
        output_names=names,
        # The decoder emits one token per step, but the same graph has to serve a
        # whole-sequence hidden state too.
        dynamic_axes={"hidden": {1: "seq_len"}, **{name: {1: "seq_len"} for name in names}},
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )
    return path_out
