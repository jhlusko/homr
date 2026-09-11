"""How the syllable recogniser reads *scans*, which nothing has measured.

Its recorded numbers -- CER 11.6%, seen 88.7%, unseen 79.0% -- come from `lyric_crops`,
which renders the OpenScore Lieder corpus through MuseScore. RUNLOG 27.38 already called
that the easy case: "synthetic rendering is not a different domain so much as the easy
case... A lyric recogniser trained on rendered text would overstate by more" than the 25
points the notation heads lost crossing to scanned.

`lieder_vocal_text` is the same corpus photographed rather than rendered: OCR-confirmed
syllables with boxes on real IMSLP pages. So it is the held-out set that measurement
needs, and it is the domain the detector actually produces boxes in.

Only exact OCR matches are used (`matched_fraction == 1.0`): a partial match is not a
label, and scoring against one would count the recogniser wrong for being right.
"""

import json
import os
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import torch

from training.architecture.ocr.crnn import CRNN, IMAGE_HEIGHT, Alphabet

VOCAL = Path(
    os.environ.get(
        "LIEDER_VOCAL_TEXT",
        "/home/jhlusko/workspace/homr-artifacts/datasets/lieder_vocal_text",
    )
)
CKPT = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/jhlusko/workspace/homr-artifacts/models/ocr/crnn-h48.pth"
)
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 2000


def cer(truth, pred):
    if not truth:
        return 0.0 if not pred else 1.0
    d = range(len(pred) + 1)
    for i, t in enumerate(truth, 1):
        nd = [i]
        for j, p in enumerate(pred, 1):
            nd.append(min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (t != p)))
        d = nd
    return d[-1] / len(truth)


blob = torch.load(CKPT, map_location="cpu", weights_only=False)
alphabet = Alphabet(blob["alphabet"])
model = CRNN(len(alphabet))
model.load_state_dict(blob["model"])
model.eval()
print(f"{CKPT.name}: alphabet {len(alphabet)-1} chars")

samples = []
for path in sorted(VOCAL.glob("*.json")):
    data = json.loads(path.read_text())
    for m in data["matches"]:
        if m["kind"] != "lyric" or m.get("matched_fraction", 0) < 1.0:
            continue
        img = VOCAL / "pages" / data["score_id"] / m["page_image"]
        if img.exists():
            samples.append((img, m["box"], m["text"], data["score_id"]))
    if len(samples) >= LIMIT:
        break
samples = samples[:LIMIT]
print(f"{len(samples)} exact-match syllables on real scans, {len({s[3] for s in samples})} scores")

pages, skipped = {}, 0
examples = []
# Split by label length: the recogniser was trained on syllables (median 3 chars), and a
# box holding a whole phrase is a different task, not a harder instance of the same one.
buckets = {"syllable (<=7 chars)": [0, 0.0, 0], "phrase (>7)": [0, 0.0, 0]}
for img_path, box, text, _ in samples:
    if img_path not in pages:
        pages.clear()
        pages[img_path] = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    page = pages[img_path]
    if page is None:
        skipped += 1
        continue
    x, y, w, h = box["left"], box["top"], box["width"], box["height"]
    crop = page[max(0, y) : y + h, max(0, x) : x + w]
    if crop.size == 0 or crop.shape[0] < 4:
        skipped += 1
        continue
    w2 = max(8, int(round(crop.shape[1] * IMAGE_HEIGHT / max(1, crop.shape[0]))))
    resized = cv2.resize(crop, (w2, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)[None, None]
    with torch.no_grad():
        logits = model(tensor)
    pred = alphabet.decode(logits.argmax(dim=-1)[:, 0].tolist())
    truth = unicodedata.normalize("NFC", text)
    key = "syllable (<=7 chars)" if len(truth) <= 7 else "phrase (>7)"
    buckets[key][0] += pred == truth
    buckets[key][1] += cer(truth, pred)
    buckets[key][2] += 1
    if len(examples) < 12:
        examples.append((truth, pred))

print("")
print("on real scans:")
for name, (hits, cer_sum, count) in buckets.items():
    if not count:
        continue
    print(f"  {name:22s} n={count:5d}  exact {hits/count:6.1%}  CER {cer_sum/count:6.1%}")
print("")
print("first predictions (truth -> read):")
for t, pr in examples:
    print(f"   {t!r:<18} -> {pr!r}")
print("")
print("rendered (recorded): exact ~88.7 pct seen / 79.0 pct unseen, CER 11.6 pct")
