import torch

from training.ocr.finetune_direction_head import HEAD_BIAS, HEAD_WEIGHT, remap_state


def test_e4_head_remap_preserves_dynamic_and_other_non_direction_filters():
    weights = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2, 1, 1)
    bias = torch.arange(8, dtype=torch.float32)
    result = remap_state({HEAD_WEIGHT: weights, HEAD_BIAS: bias, "trunk": torch.ones(1)})

    for destination, source in ((0, 0), (1, 1), (2, 2), (4, 5), (5, 7)):
        assert torch.equal(result[HEAD_WEIGHT][destination], weights[source])
        assert result[HEAD_BIAS][destination] == bias[source]
    assert torch.equal(result[HEAD_WEIGHT][3], weights[[3, 4, 6]].mean(dim=0))
    assert result[HEAD_BIAS][3] == bias[[3, 4, 6]].mean()
    assert torch.equal(result["trunk"], torch.ones(1))
