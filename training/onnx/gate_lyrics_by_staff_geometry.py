"""Exploratory scan-only staff-gap filtering and same-row OCR recovery."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from training.onnx.qualify_joint_export import sha, write


def staff_groups(image):
    binary = (image < 130).astype('uint8')
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                 np.ones((1, max(1, image.shape[1] // 8)), 'uint8'))
    ys = np.where(horizontal.sum(axis=1) > image.shape[1] * .25)[0]
    runs = []
    for y in ys:
        if not runs or y > runs[-1][-1] + 1:
            runs.append([int(y)])
        else:
            runs[-1].append(int(y))
    lines = [sum(run) / len(run) for run in runs]
    groups = []
    for y in lines:
        if not groups or y - groups[-1][-1] > 35:
            groups.append([y])
        else:
            groups[-1].append(y)
    valid = [g for g in groups if len(g) == 5 and max(np.diff(g)) / min(np.diff(g)) < 1.25]
    return valid, [g for g in groups if g not in valid]


def gap_for_box(box, staves):
    y = sum(p[1] for p in box) / len(box)
    for i, (first, following) in enumerate(zip(staves, staves[1:])):
        spacing = (first[-1] - first[0]) / 4
        if first[-1] + spacing / 2 < y < following[0] - spacing / 2:
            return i
    return None


def same_row(first, second):
    a, b = sorted(p[1] for p in first), sorted(p[1] for p in second)
    height = min(a[-1] - a[0], b[-1] - b[0])
    return height > 0 and min(a[-1], b[-1]) - max(a[0], b[0]) >= height / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pages', 'gate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Output must be new')
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True)
    summaries = []
    for variant in ('staff-gaps', 'staff-gaps-row-recovery'):
        (args.out / variant).mkdir()
    for page in json.loads(args.pages.read_text()):
        if page['partition'] != 'calibration' or sha(page['image']) != page['image_sha256']:
            raise ValueError('Calibration identity mismatch')
        image = cv2.imread(page['image'], cv2.IMREAD_GRAYSCALE)
        staves, rejected = staff_groups(image)
        if len(staves) < 2:
            raise ValueError('Insufficient staff geometry')
        source = args.gate / (page['score'] + '.json')
        data = json.loads(source.read_text())
        if data['image_sha256'] != page['image_sha256']:
            raise ValueError('Gate image mismatch')
        records = [dict(r, previously_selected=True) for r in data['all_page_text']]
        records += [dict(r, previously_selected=False) for r in data['rejected_page_text']]
        gaps = [gap_for_box(r['box'], staves) for r in records]
        seed = {i for i, r in enumerate(records) if r['previously_selected'] and gaps[i] is not None}
        recovered = set(seed)
        changed = True
        while changed:
            changed = False
            for i, record in enumerate(records):
                if gaps[i] is None or i in recovered:
                    continue
                if any(gaps[i] == gaps[j] and same_row(record['box'], records[j]['box'])
                       for j in recovered):
                    recovered.add(i)
                    changed = True
        for variant, selected in (('staff-gaps', seed), ('staff-gaps-row-recovery', recovered)):
            result = dict(data, all_page_text=[r for i, r in enumerate(records) if i in selected],
                          rejected_page_text=[r for i, r in enumerate(records) if i not in selected],
                          staff_geometry={'five_line_groups': staves, 'excluded_line_groups': rejected},
                          selection={'variant': variant, 'parent': str(source),
                                     'parent_sha256': sha(source),
                                     'source_or_reference_rows_used': False,
                                     'scope': 'Exploratory; designed after calibration review; no voice ownership'})
            write(args.out / variant / (page['score'] + '.json'), result)
            summaries.append({'score': page['score'], 'variant': variant, 'selected': len(selected),
                              'recovered': len(selected - seed), 'staves': len(staves),
                              'excluded_groups': len(rejected)})
    write(args.out / 'receipt.json', {'summary': summaries, 'source_sha256': sha(Path(__file__)),
          'pages_sha256': sha(args.pages), 'scope': 'Exploratory calibration only. Fixed scan '
          'morphology, five-line groups and half-height row overlap; no text matching/GT input. '
          'Staff gaps do not identify the vocal voice; not a production policy.'})
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
