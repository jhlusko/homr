import json

import pytest
import torch

from training.architecture.transformer.structured_heads import StructuredNotationHeads
from training.architecture.transformer.structured_losses import IGNORE_INDEX
from training.transformer.synthetic_head_capacity import (
    balanced_indices, class_report, fit_step, image_controls, load_state,
    pools_for, qualify, save_state,
)


def test_image_control_keeps_family_font_and_tokens_identical():
    def row(name, family, token):
        return dict(id=name, family=family, font='Leland', files={name+'.tokens': token})
    rows = [row('a', 'one', 'x'), row('b', 'one', 'x'), row('c', 'one', 'y'),
            row('d', 'two', 'x')]
    assert image_controls(rows) == {'a': 'b', 'b': 'a'}


def test_balanced_pools_exclude_masked_notes_and_heldout_families():
    records = [dict(split='train', family='a', hidden=torch.tensor([[1.], [2.], [3.]]),
                    targets={'stem.direction': torch.tensor([1, IGNORE_INDEX, 2])}),
               dict(split='valid', family='b', hidden=torch.tensor([[9.]]),
                    targets={'stem.direction': torch.tensor([1])})]
    pool = pools_for(records, ['stem.direction'], 'cpu')['stem.direction']
    assert pool['x'].flatten().tolist() == [1., 3.]
    assert pool['families'] == ['a', 'a']
    picked = balanced_indices(pool, 1000, torch.Generator().manual_seed(4))
    counts = torch.bincount(pool['y'][picked])
    assert 450 < int(counts[1]) < 550


def test_resume_reproduces_linear_updates_exactly_and_rejects_changed_identity(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    heads = StructuredNotationHeads(4, 4, 2)
    x = torch.randn(20, 4)
    y = (x[:, 0] > 0).long() + 1
    pool = {'stem.direction': dict(x=x, y=y, classes={c: (y == c).nonzero().flatten() for c in (1, 2)})}
    opt = torch.optim.Adam(heads.stem.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(6)
    for _ in range(3): fit_step(heads, pool, opt, 8, rng)
    save_state(tmp_path/'resume.pt', heads, opt, rng, 3, {'test': 1})
    for _ in range(4): fit_step(heads, pool, opt, 8, rng)
    resumed = StructuredNotationHeads(4, 4, 2)
    resumed_opt = torch.optim.Adam(resumed.stem.parameters(), lr=.001)
    resumed_rng = torch.Generator()
    assert load_state(tmp_path/'resume.pt', resumed, resumed_opt, resumed_rng, {'test': 1}) == 3
    for _ in range(4): fit_step(resumed, pool, resumed_opt, 8, resumed_rng)
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in heads.state_dict().items())
    with pytest.raises(ValueError, match='identity'):
        load_state(tmp_path/'resume.pt', resumed, resumed_opt, resumed_rng, {'test': 2})


def test_unsupported_classes_do_not_inflate_macro_f1():
    report = class_report(torch.tensor([1, 2, 0]), torch.tensor([1, 2, IGNORE_INDEX]), 4)
    assert report['support'] == 2 and report['macro_f1'] == 1
    assert report['classes']['0']['support'] == 0


def test_unapproved_generator_cannot_be_used(tmp_path):
    (tmp_path/'identity.json').write_text(json.dumps({'schema': 'v4'}))
    approval = tmp_path/'approval.json'
    approval.write_text(json.dumps({'approved': False}))
    with pytest.raises(ValueError, match='qualification'):
        qualify(tmp_path, approval)
