import json
from collections import Counter

import pytest
import torch

from training.transformer.synthetic_head_transfer import (
    collate_fixed, compare, make_schedules, quotas,
)
from training.architecture.transformer.structured_losses import IGNORE_INDEX


def test_matched_multiset_and_fixed_budget_with_heldout_families_excluded():
    real=[dict(source=s) for s in ('ossq','pdmx','lieder') for _ in range(4)]
    synthetic=[dict(family=f'{kind}-{family}',kind=kind,variant=v,font=font,split='train')
               for kind in ('stems','ties','slurs','beams') for family in range(3)
               for v in range(5 if kind=='slurs' else 4) for font in ('Leland','Bravura')]
    heldout=len(synthetic)
    synthetic.append(dict(family='heldout',kind='slurs',variant=0,font='Bravura',split='valid'))
    schedules=make_schedules(real,synthetic,123,stages=((5,.75),(10,.5),(10,.25),(15,.1)))
    assert all(len(s)==320 for s in schedules.values())
    fingerprint=lambda rows:Counter(json.dumps(r,sort_keys=True) for r in rows)
    assert fingerprint(schedules['unordered'])==fingerprint(schedules['curriculum'])
    assert schedules['unordered']!=schedules['curriculum']
    assert all(d['kind']=='real' for d in schedules['real_only'])
    assert all(d['index']!=heldout for d in schedules['curriculum'] if d['kind']=='synthetic')
    assert len({d['draw_id'] for d in schedules['curriculum']})==320
    assert sum(d['kind']=='synthetic' for d in schedules['curriculum'])==102


def test_real_source_quotas_are_exact_and_stable():
    assert quotas(43600,dict(ossq=31496,pdmx=5701,lieder=3093)) == dict(ossq=34084,pdmx=6169,lieder=3347)
    assert sum(quotas(17,dict(ossq=7,pdmx=2,lieder=1)).values())==17


def test_dynamic_padding_preserves_target_mask_and_next_token_alignment():
    rows=[dict(inputs=torch.zeros(1,2,2),rhythms=torch.tensor([1,4,2]),mask=torch.tensor([True]*3),
               **{'stem.direction':torch.tensor([IGNORE_INDEX,1,IGNORE_INDEX])}),
          dict(inputs=torch.zeros(1,2,2),rhythms=torch.tensor([1,4,4,2]),mask=torch.tensor([True]*4),
               **{'stem.direction':torch.tensor([IGNORE_INDEX,2,1,IGNORE_INDEX])})]
    batch=collate_fixed(rows)
    assert batch['mask'][0].tolist()==[True,True,True,False]
    assert batch['stem.direction'][0].tolist()==[IGNORE_INDEX,1,IGNORE_INDEX,IGNORE_INDEX]
    assert batch['stem.direction'][0,1:].tolist()==[1,IGNORE_INDEX,IGNORE_INDEX]


def report(value):
    cls={'support':200,'recall':.8}
    return dict(reference_sha256='same',pixels_sha256='same',
                beam_levels={str(i):dict(macro_f1=value,support=200) for i in range(1,5)},
                hooks=dict(f1=value,support=200),ties=dict(macro_f1=value,support=200),
                slur_spans=dict(f1=value,support=200),slur_sides=dict(macro_f1=value,support=200),
                advance_nontrivial=dict(macro_f1=value,support=200),stems=dict(direction_only=value,macro_f1=value),
                directional_stem_support=200,exact_beam_vector=dict(rate=value,total=200),
                class_details=dict(stems={'up':cls},ties={'start':cls},slur_sides={'above':cls},beams={}),
                slur_span_counts=dict(fp=0,support=200))


def test_guardrails_reject_hidden_inventions_even_when_macro_scores_improve():
    control={s:report(.5) for s in ('ossq','pdmx','lieder')}
    candidate={s:report(.52) for s in control}
    assert compare(control,candidate)['eligible']
    candidate['pdmx']['slur_span_counts']['fp']=3
    assert not compare(control,candidate)['eligible']
    candidate['pdmx']['slur_span_counts']['fp']=0
    candidate['pdmx']['pixels_sha256']='changed'
    with pytest.raises(ValueError,match='pixels'):compare(control,candidate)
