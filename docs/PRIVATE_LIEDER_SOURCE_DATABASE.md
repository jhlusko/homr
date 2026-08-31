---
pretty_name: Private Lieder source database
license: other
tags:
- omr
- sheet-music
- provenance
- private
---

# Private Lieder source database

This private repository is the provenance and reconstruction database for HOMR's
boundary-safe OpenScore Lieder corpus. It records **all 215 exact IMSLP source file
IDs** represented in the v4 boundary-safe manifests (3,922 staff-crop rows), including
the 25 conservative public-release candidates, 64 exclusions, and 126 records awaiting
manual review.

## Contents

- `provenance/imslp_rights_audit_v4.csv` and `.jsonl`: one row per exact IMSLP file,
  with work, file name, uploader, edition evidence, stated IMSLP copyright/image type,
  source URL, audit status, and crop count.
- `manifests/`: complete score ID list plus the v4 boundary-safe source-disjoint train
  and validation indexes.
- `public-candidate/`: the 25-ID public-candidate allow-list and filtered manifests.
- `docs/`: the release audit and Hugging Face publication plan.
- `scripts/`: the exact IMSLP-record audit implementation.

It does not yet contain scan PDFs, rendered pages, staff crops, or model-derived labels.
Those payloads remain access-controlled project artifacts until a separate transfer
manifest and provenance review are approved.

## Rights and access boundary

OpenScore Lieder symbolic transcriptions are CC0 according to their source project.
The images originate in individual IMSLP uploads and their status is file-specific.
Private hosting is not a licence grant. Do not switch this repository public or copy its
scan-derived payload into a public repository by default.

The only potential public subset is documented in `public-candidate/`: each of its 25
exact file records was retrieved on 2026-08-31 and states both `Copyright=Public Domain`
and `Image Type=Normal Scan`. That remains a candidate pending final human review.

## Reproduce the audit

Run the script with the same v4 indexes and an OpenScore Lieder `scores.yaml` cache:

```bash
python3 -m training.omr_datasets.audit_lieder_imslp_rights \
  --index lieder-rebuild/imslp_train_index_v4_boundary_safe.txt \
  --index lieder-rebuild/imslp_val_index_v4_boundary_safe.txt \
  --scores-yaml scores.yaml --cache imslp-api \
  --out-jsonl imslp_rights_audit_v4.jsonl \
  --out-csv imslp_rights_audit_v4.csv
```

The audit records exclusions rather than inferring a licence from a composer, title, or
work-level IMSLP page.
