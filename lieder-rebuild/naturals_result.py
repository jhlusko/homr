"""Score N-recall and lift-branch accuracy: did the model learn to predict naturals?"""
import json, sys
from pathlib import Path
PAD = "\x00"
def real(s): return [t for t in s if not t.startswith(PAD)]

for path in sys.argv[1:]:
    n_ref = n_hit = n_pred = 0
    lift_tot = lift_hit = 0
    staves = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        want, got = real(row.get("lift_reference", [])), real(row.get("lift_predicted", []))
        if not want or not got: continue
        staves += 1
        w = min(len(want), len(got))
        for i in range(w):
            lift_tot += 1
            lift_hit += want[i] == got[i]
            if want[i] == "N":
                n_ref += 1
                n_hit += got[i] == "N"
            if got[i] == "N":
                n_pred += 1
    print(f"{Path(path).name}: staves={staves}")
    print(f"  lift accuracy    : {100*lift_hit/max(lift_tot,1):.2f}%  ({lift_hit}/{lift_tot})")
    print(f"  N in reference   : {n_ref}")
    print(f"  N recall         : {100*n_hit/max(n_ref,1):.1f}%  ({n_hit}/{n_ref})")
    print(f"  N predicted total: {n_pred}  (precision would need matching, shown for scale)")
