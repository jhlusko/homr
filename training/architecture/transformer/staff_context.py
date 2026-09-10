"""
§4/§7.4's Stage C: a learned, masked, variable-length cross-staff context adapter.

Two of this project's own prior interventions both succeeded by conditioning on
*decoded content* (Phase 1's rerank: cumulative barline positions from the rhythm
head's own output; phase23's cross-staff coherence loss: ground-truth duration
curves), never raw visual features - see `DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.4
for the full evidence and the resulting design decision. This module is the
"jointly-trained, learned" version of that same signal: each staff of a system is
decoded once (cheaply, greedily), the resulting per-staff hidden states are combined
here into per-staff context vectors, and a second decode pass uses them.

Zero-initialized gate, `ProfileContextEmbedding`'s own convention (`training/
architecture/transformer/profile_context.py`) - a raw learned scalar multiplied
directly, not a sigmoid (`sigmoid(0) = 0.5`, not zero, so a sigmoid gate could not
reproduce the exact baseline at initialization the way this needs to). Attaching
this module, and even training with it enabled, must leave every staff's decode
identical to the shared decoder run alone until the gate moves off zero.
"""

import torch
from torch import nn

#: A generous upper bound on staves in one system (piano trio/quartet/grand staff
#: writing rarely exceeds this) - positions beyond it are masked out the same as any
#: other padding, not a hard limit on what a real page could contain, since the
#: attention itself is already correct for any N <= this bound.
MAX_STAVES_PER_SYSTEM = 12


class StaffContextTransformer(nn.Module):
    def __init__(self, dim: int, heads: int = 8, ff_dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.staff_position_emb = nn.Embedding(MAX_STAVES_PER_SYSTEM, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim or dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.projection = nn.Linear(dim, dim)
        # Zero-initialized per the module docstring above - the unconditioned path
        # must be bit-identical at initialization regardless of what the encoder and
        # projection compute.
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.staff_position_emb.weight, std=0.02)

    def forward(self, staff_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`staff_hidden`: `(batch, N, dim)`, one pooled hidden-state summary per
        staff of a system, `N` variable (padded to a fixed width by the caller).
        `mask`: `(batch, N)`, `True` for a real staff, `False` for padding - a
        system with only one staff (no real siblings) still runs correctly:
        self-attention over a single unmasked position simply attends to itself,
        the same degenerate case a real one-part page (or a page cropped down to
        one visible staff) produces naturally, not a special case to detect.

        Returns `(batch, N, dim)`, the *gated, projected* context bias ready to add
        directly to each staff's own decoder input - callers never see the raw
        (ungated) encoder output, the same "own the gate" convention
        `ProfileContextEmbedding` uses.
        """
        batch, n, _ = staff_hidden.shape
        positions = torch.arange(n, device=staff_hidden.device).clamp(max=MAX_STAVES_PER_SYSTEM - 1)
        x = staff_hidden + self.staff_position_emb(positions).unsqueeze(0)

        # nn.TransformerEncoder's key_padding_mask is True at positions to *ignore* -
        # the inverse of this module's own "True means real" convention.
        context = self.encoder(x, src_key_padding_mask=~mask)
        # A staff with no real siblings at all (mask all-False in its own row, e.g. a
        # fully-padded batch slot) can produce NaN from an all-masked softmax -
        # zeroed explicitly rather than trusted to the caller to never construct
        # such a slot.
        context = torch.nan_to_num(context, nan=0.0)

        return self.gate * self.projection(context)
