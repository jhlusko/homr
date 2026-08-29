import sys, json, collections
from pathlib import Path
sys.path.insert(0, "/workspace/b0/homr")
from training.transformer.training_vocabulary import read_tokens
from training.omr_datasets.audit_label_consistency import simultaneity_groups
rows = [l.split(",",1) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
per_score = collections.Counter(); pairs_per_score = collections.Counter()
surplus = []
tot_notes = tot_extra = 0
bar_hits = 0; bars_tot = 0
for image, tokens in rows:
    stem = Path(image).stem; score = stem.split("-sys")[0]
    pairs_per_score[score] += 1
    try: sym = read_tokens(tokens)
    except Exception: continue
    notes = [s for s in sym if s.rhythm.startswith("note_")]
    extra = 0
    for g in simultaneity_groups(sym):
        for pos in ("upper","lower"):
            k=[(s.rhythm,s.pitch,s.lift) for s in g if s.position==pos and s.rhythm.startswith("note_")]
            extra += len(k)-len(set(k))
    tot_notes += len(notes); tot_extra += extra
    if extra:
        per_score[score] += 1
        surplus.append(extra/max(len(notes),1))
surplus.sort()
print(json.dumps({
 "total_notes": tot_notes, "total_duplicate_notes": tot_extra,
 "rate_of_all_notes": round(100*tot_extra/tot_notes,3),
 "affected_pairs": len(surplus), "total_pairs": len(rows),
 "scores_with_any": len(per_score), "scores_total": len(pairs_per_score),
 "surplus_median_pct": round(100*surplus[len(surplus)//2],1) if surplus else 0,
 "surplus_p90_pct": round(100*surplus[int(len(surplus)*0.9)],1) if surplus else 0,
 "top_scores": [(s, per_score[s], pairs_per_score[s]) for s,_ in per_score.most_common(12)],
}, indent=2))
