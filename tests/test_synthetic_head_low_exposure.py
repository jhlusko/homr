from collections import Counter
import json
from training.transformer.synthetic_head_low_exposure import schedules


def test_low_dose_preserves_real_draw_sequence_and_matches_candidate_multisets():
    plan=dict(real=[dict(source=s) for s in ('ossq','pdmx','lieder')],
              synthetic=[dict(split='train',family=str(i),font='Bravura',variant=0,kind='slurs') for i in range(4)],
              schedules=dict(real_only=[dict(draw_id=i,kind='real',index=i%3,stage=0,pixel_seed=i) for i in range(64000)]))
    got=schedules(plan)
    assert all(len(v)==67200 for v in got.values())
    assert sum(d['kind']=='synthetic' for d in got['curriculum'])==3200
    assert [d['index'] for d in got['curriculum'] if d['kind']=='real']==[d['index'] for d in plan['schedules']['real_only']]
    assert [d['pixel_seed'] for d in got['curriculum'] if d['kind']=='real']==list(range(64000))
    key=lambda rows:Counter(json.dumps(d,sort_keys=True) for d in rows)
    assert key(got['unordered'])==key(got['curriculum'])
    assert got['real_only'][:64000]==plan['schedules']['real_only']
    assert all(d['kind']=='real' for d in got['real_only'])
