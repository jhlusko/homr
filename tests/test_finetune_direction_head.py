import torch
import pytest
from torch.utils.data import WeightedRandomSampler

from training.ocr.finetune_direction_head import (
    HEAD_BIAS, HEAD_WEIGHT, load_trainer_state, remap_state, save_trainer_state,
)


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


def test_atomic_trainer_state_restores_weights_optimizer_and_sampler(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sampler = WeightedRandomSampler([1.0, 1.0], 4, generator=torch.Generator().manual_seed(7))
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "trainer-state.pth"
    save_trainer_state(path, model, optimizer, sampler, [{"epoch": 1}], {"data": "fixed"})
    expected_draw = list(sampler)
    with torch.no_grad():
        model.weight.zero_()
    sampler.generator.manual_seed(99)
    assert load_trainer_state(path, model, optimizer, sampler, {"data": "fixed"}) == [
        {"epoch": 1}
    ]
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original.items())
    assert list(sampler) == expected_draw
    with pytest.raises(ValueError, match="digests changed"):
        load_trainer_state(path, model, optimizer, sampler, {"data": "different"})
