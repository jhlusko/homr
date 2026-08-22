import json
import os
from typing import Any

from homr.transformer.structured_notation import (
    TRAINED_BEAM_LEVELS,
    TRAINED_SLUR_SLOTS,
)
from homr.transformer.vocabulary import Vocabulary

workspace = os.path.join(os.path.dirname(__file__))
root_dir = os.getcwd()


class FilePaths:
    def __init__(self) -> None:
        model_name = "pytorch_model_426-b6fd20809a8dcaf10dfd39a4ca4f64c6f056e644"
        self.encoder_path = os.path.join(
            workspace,
            f"encoder_{model_name}.onnx",
        )  # noqa: E501
        self.decoder_path = os.path.join(
            workspace,
            f"decoder_{model_name}.onnx",
        )  # noqa: E501

        self.encoder_path_fp16 = os.path.join(
            workspace,
            f"encoder_{model_name}_fp16.onnx",
        )  # noqa: E501
        self.decoder_path_fp16 = os.path.join(
            workspace,
            f"decoder_{model_name}_fp16.onnx",
        )  # noqa: E501

        self.checkpoint = os.path.join(
            root_dir,
            "training",
            "architecture",
            "transformer",
            f"{model_name}.pth",
        )

        self.rhythmtokenizer = os.path.join(workspace, "tokenizer_rhythm.json")
        self.lifttokenizer = os.path.join(workspace, "tokenizer_lift.json")
        self.pitchtokenizer = os.path.join(workspace, "tokenizer_pitch.json")
        self.notetokenizer = os.path.join(workspace, "tokenizer_note.json")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "rhythmtokenizer": self.rhythmtokenizer,
            "lifttokenizer": self.lifttokenizer,
            "pitchtokenizer": self.pitchtokenizer,
            "notetokenizer": self.notetokenizer,
        }

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class DecoderArgs:
    def __init__(self) -> None:
        self.attn_on_attn = True
        self.cross_attend = True
        self.ff_glu = True
        self.rel_pos_bias = False
        self.use_scalenorm = False
        self.attn_dropout = 0.1
        self.ff_dropout = 0.1
        self.layer_dropout = 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "attn_on_attn": self.attn_on_attn,
            "cross_attend": self.cross_attend,
            "ff_glu": self.ff_glu,
            "rel_pos_bias": self.rel_pos_bias,
            "use_scalenorm": self.use_scalenorm,
            "attn_dropout": self.attn_dropout,
            "ff_dropout": self.ff_dropout,
            "layer_dropout": self.layer_dropout,
        }

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class Config:
    def __init__(self) -> None:
        self.vocab = Vocabulary()
        self.filepaths = FilePaths()
        self.channels = 1
        self.patch_size = 16
        self.max_height = 256
        self.max_width = 1280
        self.max_seq_len = 608
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.nonote_token = 0
        self.num_rhythm_tokens = len(self.vocab.rhythm)
        self.num_pitch_tokens = len(self.vocab.pitch)
        self.num_lift_tokens = len(self.vocab.lift)
        self.num_articulation_tokens = len(self.vocab.articulation)
        self.num_slur_tokens = len(self.vocab.slur)
        self.num_position_tokens = len(self.vocab.position)

        # Structured notation heads (beams, stem direction, slurs). Off by default: they
        # are output-only additions, and a checkpoint trained without them must keep
        # loading and behaving exactly as before.
        self.enable_structured_heads = False
        self.structured_beam_levels = TRAINED_BEAM_LEVELS
        self.structured_slur_slots = TRAINED_SLUR_SLOTS
        # §7.2/§7.3 score-profile conditioning. Off by default, same reasoning as the
        # structured heads: a checkpoint trained without it must keep loading and
        # behaving exactly as before. Even when enabled, the embedding's own gate is
        # zero-initialized (training.architecture.transformer.profile_context), so
        # turning this on is itself still a no-op until training moves the gate.
        self.enable_profile_context = False
        # DECODER_RHYTHM_ACCURACY_DESIGN.md §7.3's ground-truth-supervised measure-
        # duration adherence loss: penalizes the rhythm head's own *expected* (softmax-
        # weighted) cumulative duration for diverging from the ground-truth cumulative
        # duration at each true barline position - a differentiable analogue of Stage
        # A's `check_measure_durations`/`_cumulative_barline_positions`, targeting the
        # single largest Stage A finding (`barline_position_mismatch`) directly at
        # training time. 0.0 (off) preserves the existing loss exactly - the same
        # "zero means no effect, safe to land ahead of being trained" discipline
        # `profile_context`'s own gate uses, expressed as a loss weight instead of a
        # module gate since this changes the *objective* itself, not a decoder input.
        self.duration_adherence_weight = 0.0
        # DECODER_RHYTHM_ACCURACY_DESIGN.md §7.3's loss brainstorm item 2: penalizes
        # the rhythm head's own predicted cumulative duration for diverging from its
        # *system's* ground truth (the median across every sibling part, not just this
        # staff's own label) at each of this staff's own barlines - a cheaper training-
        # time alternative to §4 Stage C's learned cross-staff adapter, worth measuring
        # before committing to that larger build. 0.0 (off) preserves the existing
        # loss exactly, same discipline as duration_adherence_weight above.
        self.cross_staff_coherence_weight = 0.0
        # §4/§7.4 Stage C: the learned cross-staff adapter itself (StaffContextTransformer),
        # as opposed to cross_staff_coherence_weight's cheaper loss-only alternative above.
        # Off by default, same reasoning as enable_profile_context: a checkpoint trained
        # without it must keep loading and behaving exactly as before, and its own gate is
        # zero-initialized, so enabling it is itself still a no-op until training moves the
        # gate (train_staff_context.py).
        self.enable_staff_context = False
        self.encoder_structure = "convnext"
        self.encoder_depth = 8
        self.backbone_layers = [3, 4, 6, 3]
        self.encoder_dim = 512
        # encoder_h_dim balances how many dimensions the
        # horizontal vs vertical embeddings get
        self.encoder_h_dim = self.encoder_dim // 3
        self.encoder_heads = 8
        self.decoder_dim = self.encoder_dim
        self.decoder_depth = 8
        self.decoder_heads = 8
        self.decoder_args = DecoderArgs()
        self.lift_vocab = self.vocab.lift
        self.pitch_vocab = self.vocab.pitch
        self.rhythm_vocab = self.vocab.rhythm
        self.articulation_vocab = self.vocab.articulation
        self.slur_vocab = self.vocab.slur
        self.position_vocab = self.vocab.position
        self.use_gpu_inference = True
        # Opt-in: run the encoder on the Apple GPU via CoreML (MLProgram). Off
        # by default because compiling the MLProgram costs 26-60 s at session
        # creation, so it only pays off across many images. Set via the
        # --coreml-encoder CLI flag.
        self.use_coreml_encoder = False

        # Scheduled Sampling parameters
        self.scheduled_sampling_start_prob = 1.0
        self.scheduled_sampling_end_prob = 0.4
        self.scheduled_sampling_decay_steps = 45000

    def to_dict(self) -> dict[str, Any]:
        return {
            "filepaths": self.filepaths.to_dict(),
            "channels": self.channels,
            "patch_size": self.patch_size,
            "max_height": self.max_height,
            "max_width": self.max_width,
            "max_seq_len": self.max_seq_len,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "nonote_token": self.nonote_token,
            "encoder_structure": self.encoder_structure,
            "encoder_depth": self.encoder_depth,
            "backbone_layers": self.backbone_layers,
            "encoder_dim": self.encoder_dim,
            "encoder_heads": self.encoder_heads,
            "num_rhythm_tokens": self.num_rhythm_tokens,
            "decoder_dim": self.decoder_dim,
            "decoder_depth": self.decoder_depth,
            "decoder_heads": self.decoder_heads,
            "decoder_args": self.decoder_args.to_dict(),
        }

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# Initialize the Config class
default_config = Config()
