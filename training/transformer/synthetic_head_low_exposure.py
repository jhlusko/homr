"""Prepare a coverage-preserving low-dose plan for the unchanged transfer runner."""
from collections import Counter
import argparse
import copy
import json
from pathlib import Path
import random
from training.transformer.synthetic_head_transfer import draw_real, draw_synthetic
from training.transformer.synthetic_head_capacity import sha, write_json


def schedules(plan, seed=20260928):
    real = copy.deepcopy(plan['schedules']['real_only'])
    if len(real) != 64000 or any(d['kind'] != 'real' for d in real):
        raise ValueError('Expected the original 64,000 real draws')
    rng = random.Random(seed + 50)
    # Keep all real draws in their original order; interleave 10%, 5%, then 2%
    # additional draws relative to the real count in each stage.
    ordered = []
    start = 0
    for stage, (count, extra) in enumerate(((16000,1600),(16000,800),(32000,800))):
        real_stage = real[start:start+count]; start += count
        synthetic = draw_synthetic(plan['synthetic'],extra,3,rng)
        positions = sorted(rng.sample(range(count),extra))
        inserts = dict(zip(positions,synthetic,strict=True))
        for offset, draw in enumerate(real_stage):
            ordered.append(draw)
            if offset in inserts:
                ordered.append(dict(kind='synthetic',index=inserts[offset],stage=stage,
                                    pixel_seed=rng.randrange(2**32)))
    for i,d in enumerate(ordered):d['draw_id']=i
    unordered=copy.deepcopy(ordered);random.Random(seed+51).shuffle(unordered)
    control=copy.deepcopy(real)
    for i in draw_real(plan['real'],3200,random.Random(seed+52)):
        control.append(dict(kind='real',index=i,stage=3,pixel_seed=rng.randrange(2**32)))
    for i,d in enumerate(control):d['draw_id']=i
    return dict(real_only=control,unordered=unordered,curriculum=ordered)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-plan',required=True,type=Path)
    p.add_argument('--run',required=True,type=Path)
    a=p.parse_args()
    if a.run.exists() and any(a.run.iterdir()):raise ValueError('Fresh output required')
    plan=json.loads(a.parent_plan.read_text())
    result=schedules(plan)
    identity=copy.deepcopy(plan['identity'])
    identity.update(updates=8400,parent_plan_sha256=sha(a.parent_plan),
                    experiment='low-dose-real-coverage-preserved-20260928',
                    synthetic_fraction=3200/67200,real_draws_preserved=64000,
                    stages=[[16000,1600],[16000,800],[32000,800]],
                    synthetic_stage_pools='all qualified training families/fonts at every stage')
    for arm,draws in result.items():
        identity[arm]=dict(draws=len(draws),source_counts=dict(Counter(
            plan['real'][d['index']]['source'] if d['kind']=='real' else 'synthetic' for d in draws)))
    write_json(a.run/'plan.json',dict(identity=identity,real=plan['real'],synthetic=plan['synthetic'],schedules=result))

if __name__=='__main__':main()
