from typing import Any

import torch
from torch import nn

from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from training.architecture.transformer.decoder import get_decoder
from training.architecture.transformer.encoder import get_encoder


class TrOMR(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.config = config

    def eval_mode(self) -> None:
        self.decoder.eval()
        self.encoder.eval()

    def forward(
        self,
        inputs: torch.Tensor,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        slurs: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        sampling_prob: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        context = self.encoder(inputs)
        loss = self.decoder(
            rhythms=rhythms,
            pitchs=pitchs,
            lifts=lifts,
            articulations=articulations,
            slurs=slurs,
            positions=positions,
            context=context,
            mask=mask,
            sampling_prob=sampling_prob,
            **kwargs,
        )
        return loss

    @torch.no_grad()
    def generate(self, x: torch.Tensor) -> list[EncodedSymbol]:
        start_token = torch.tensor([[1]], dtype=torch.long, device=x.device)
        nonote_token = torch.tensor([[0]], dtype=torch.long, device=x.device)

        context = self.encoder(x)
        out = self.decoder.generate(start_token, nonote_token, context=context)

        return out

    def freeze_decoder(self) -> None:
        """Freeze all decoder parameters to prevent updates during training."""
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters to prevent updates during training."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_backbone(self) -> None:
        """Freeze only the encoder backbone."""
        if hasattr(self.encoder, "freeze_backbone"):
            self.encoder.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        """Unfreeze the encoder backbone."""
        if hasattr(self.encoder, "unfreeze_backbone"):
            self.encoder.unfreeze_backbone()

    def freeze_core_for_structured_heads(self) -> list[str]:
        """Train only the structured heads; freeze everything the checkpoint provided.

        This is the first experiment's whole design: if the pretrained representation
        already carries enough visual evidence for explicit beaming, stem direction and
        richer slurs, heads over a frozen core will learn them. Anything else moving
        makes the result unattributable - a gain could be the heads, or the core drifting
        to suit them.

        The existing fine-tuning path freezes most of the model and trains only the lift
        head, which is a different objective and not reusable here.

        Returns the names of the parameters left trainable, so a run can record what it
        actually trained rather than what it intended to.
        """
        if self.decoder.structured_heads is None:
            raise ValueError("no structured heads to train - set config.enable_structured_heads")
        trainable = []
        for name, param in self.named_parameters():
            train = name.startswith("decoder.structured_heads.")
            param.requires_grad = train
            if train:
                trainable.append(name)
        return trainable

    def freeze_core_for_profile_context(self) -> list[str]:
        """Train only §7.3's score-profile embedding; freeze everything the checkpoint
        provided - the same experiment shape as `freeze_core_for_structured_heads`, for
        the same reason: attributability. `ProfileContextEmbedding`'s gate starts at
        zero, so this asks a narrow question - can *some* assignment of the embedding
        tables and gate, backpropagated through the frozen network, move the model's
        own existing loss (not a new one; profile context conditions the input, it does
        not add an output) - before committing to the more expensive question of
        whether letting the core itself adapt to the signal does better.

        Returns the names of the parameters left trainable, so a run can record what it
        actually trained rather than what it intended to.
        """
        if self.decoder.profile_context is None:
            raise ValueError("no profile context to train - set config.enable_profile_context")
        trainable = []
        for name, param in self.named_parameters():
            train = name.startswith("decoder.profile_context.")
            param.requires_grad = train
            if train:
                trainable.append(name)
        return trainable

    def freeze_core_for_staff_context(self) -> list[str]:
        """Train only §4/§7.4's cross-staff `StaffContextTransformer`; freeze everything
        the checkpoint provided - the same frozen-core probe shape as
        `freeze_core_for_profile_context`, for the same reason: attributability.
        `StaffContextTransformer`'s gate starts at zero, so this asks the same narrow
        question profile_context's probe did - can *some* assignment of the module,
        backpropagated through the frozen network, move the model's own existing loss -
        before committing to letting the core itself adapt to cross-staff signal.

        Returns the names of the parameters left trainable, so a run can record what it
        actually trained rather than what it intended to.
        """
        if self.decoder.staff_context is None:
            raise ValueError("no staff context to train - set config.enable_staff_context")
        trainable = []
        for name, param in self.named_parameters():
            train = name.startswith("decoder.staff_context.")
            param.requires_grad = train
            if train:
                trainable.append(name)
        return trainable

    def unfreeze_decoder_for_profile_context(self) -> list[str]:
        """The natural next experiment after `freeze_core_for_profile_context`'s
        frozen-core probe found real signal (phase20, ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md
        §3: 10/10 epochs positive, mean delta +0.0615): let the decoder adapt to the
        signal instead of only the embedding that feeds it.

        Deliberately narrower than a full-model fine-tune: the visual encoder never sees
        profile context at all (it only processes staff image crops), so unfreezing it
        widens risk - a much bigger, harder-to-attribute change - without being the
        variable this experiment is actually testing. Freezes the encoder explicitly
        (rather than leaving it at whatever state a caller left it in) and unfreezes
        every decoder parameter, profile context included.

        Returns the names of the parameters left trainable, so a run can record what it
        actually trained rather than what it intended to.
        """
        if self.decoder.profile_context is None:
            raise ValueError("no profile context to train - set config.enable_profile_context")
        self.freeze_encoder()
        trainable = []
        for name, param in self.decoder.named_parameters():
            param.requires_grad = True
            trainable.append(f"decoder.{name}")
        return trainable

    def unfreeze_lift_decoder(self) -> None:
        for param in self.decoder.net.lift_emb.parameters():
            param.requires_grad = True
        for param in self.decoder.net.to_logits_lift.parameters():
            param.requires_grad = True


def load_model(config: Config) -> TrOMR:
    """Load model from checkpoint."""
    model = TrOMR(config)
    checkpoint_path = config.filepaths.checkpoint
    if checkpoint_path.endswith(".safetensors"):
        import safetensors  # noqa: PLC0415

        tensors = {}
        with safetensors.safe_open(checkpoint_path, framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False
        )
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model
