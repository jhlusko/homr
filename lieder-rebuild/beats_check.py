"""Does the numerator a label STATES agree with the bar durations it WRITES?

A `timeSignatureBeats_n` token is a claim about the metre that can be checked without
a model, a scan, or ground truth: n quarter-notes-worth of beats against the staff's
own modal bar. IMSLP405017-sys17-v0 states 4 while every one of its bars holds exactly
three quarters - the score changes metre and the cutter carried the earlier numerator
forward - and the renderer prefers the stated numerator, so the reviewer is shown 4/4
over music that is plainly in 3.
"""
import sys, collections, json
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, "/workspace/b0/homr")
from training.transformer.training_vocabulary import read_tokens
from training.omr_datasets.audit_label_consistency import measure_durations, is_single_staff

rows = [l.split(",",1) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
stats = collections.Counter()
examples = []
for image, tokens in rows:
    try: sym = read_tokens(tokens)
    except Exception: continue
    if not is_single_staff(sym):
        stats["skipped_grand_staff"] += 1; continue
    beats = [s.rhythm for s in sym if s.rhythm.startswith("timeSignatureBeats_")]
    denoms = [s.rhythm.split("/")[1] for s in sym if s.rhythm.startswith("timeSignature/")]
    if not beats:
        stats["no_stated_numerator"] += 1; continue
    if len(set(beats)) > 1:
        stats["numerator_changes_within_crop"] += 1; continue
    stated = int(beats[0].split("_")[1])
    denom = int(denoms[0]) if denoms else 4
    durs = measure_durations(sym)
    if len(durs) < 3:
        stats["too_few_bars"] += 1; continue
    modal = collections.Counter(durs).most_common(1)[0][0]
    implied = modal * denom          # bar length in units of 1/denom
    stats["checked"] += 1
    if implied == stated:
        stats["agree"] += 1
    else:
        stats["DISAGREE"] += 1
        if len(examples) < 12:
            examples.append({"stem": Path(image).stem, "stated": stated,
                             "implied": str(implied), "denominator": denom,
                             "modal_bar": str(modal)})
print(json.dumps({"stats": dict(stats), "examples": examples}, indent=2))
c = stats["checked"] or 1
print(f"\nstated numerator contradicts the label's own bars: "
      f"{stats['DISAGREE']}/{c} = {100*stats['DISAGREE']/c:.1f}%")
