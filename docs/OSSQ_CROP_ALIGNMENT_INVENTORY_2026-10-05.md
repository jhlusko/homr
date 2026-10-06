# Reviewed page boundaries and local OSSQ crop inventory

User review confirms the four displayed pages: Haydn sq9146376, Glazunov
sq7358579, Chaminade IMSLP154086 and Franz IMSLP89797. The Chaminade scan has
four systems with 3, 3, 2, 3 measures. MuseScore reflows this source into two
rendered pages; its line breaks do not establish scan boundaries. The final
viewer shows all rendered pages and the frozen scan-system boundary table.

The review receipt now includes the user's confirmation and hashes of the exact
HTML, complete rendered PNGs, PDFs, original scans and source files. The initial
receipt and certified packet remain preserved. The artifact-bound packet is:

`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-page-extension-20261005/certified-bound-artifacts/`

It records page/part/order/boundary review. It does not assert exhaustive note
fidelity or remove historical evaluation exposure. Lieder source part counts
are read from the actual compressed reference, rather than assumed to be four.

## Actual crop support check

The local `homr-artifacts/corpora/ossq_scanned_corrected_portable` corpus has
34,510 train crops and 3,911 valid crops across 84 scores. Both corpus indexes
agree exactly with the local label inventory and all indexed images/labels exist.
Neither reviewed OSSQ score appears in these indexes: the two reviewed pages are
evaluation controls and contribute zero training-crop alignment candidates.

All 84 indexed scores have nonempty scan CSV files, but none has local
`metadata/scanned/systemwise/*.yaml`. Presence of a CSV does not establish exact
measure boundaries; the synthetic/unaligned metadata cannot substitute for scan
metadata. The original scan and source can use different pagination. This local
legacy corpus inventory is also not the stem-restored public-v2 model index.

Evidence, source hashes and per-score counts:

`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-page-extension-20261005/local-crop-alignment-inventory-v2.json`

Reproduce with `python -m training.onnx.inventory_ossq_crop_alignment`, supplying
`--corpus`, `--ossq`, `--certified` and a new `--out` JSON path. The tool refuses
replacement, changed certified sources, malformed/duplicate indexes, missing
indexed files and mismatched index/label inventories. The preliminary directory
inventory is retained separately. One CPU thread; all work complete, ETA zero.

Eleven existing page-certification tests pass, including changed evidence,
incorrect system counts, prediction-assisted review refusal and failed-decode
accounting. Scoped formatting/lint pass. No recognition rerun is needed for this
inventory.

Next recover archived mappings for actual indexed training/validation crops and
the public-v2 build manifest. Bind crop pixels, source physical measures, part
identity and split before replaying timing sidecars or counting class support.
Do not move reviewed evaluation pages into training to fill this gap. The timing
trainer can be prepared independently, using qualified Lieder labels until OSSQ
crop mapping is recovered.

## Public-v2 pins and historical mapping recovery

Recovered the public manifest and both indexes, checked all six shard entries
against `ossq-omr-data/sources.lock.json`, and verified 35,226 train windows /
76 scores and 3,911 valid windows /9 scores with no score overlap. Neither
reviewed OSSQ score appears in these public indexes either.

The already-local validation shard matches its full published SHA-256; every
indexed image, token and notation sidecar is present (11,733 members). The
307 MiB frozen source archive also matches its pinned size and SHA-256. It
contains no generated alignment/detector metadata. The August GPU backup,
however, has sources and detector metadata for all 85 indexed scores and scanned
system metadata for 84. `sq10675759` lacks scanned-system metadata.

Evidence is local:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-mapping-recovery-20261005/`.
Receipts: `public-index-receipt.json`, `valid-shard-receipt.json`,
`archive-metadata-receipt.json`, `backup-metadata-inventory.json`.

`training.onnx.recover_ossq_archive_mapping` joins actual public index rows to
historical metadata and archived partwise pixels using eight workers. It checks
score/page/system/part identities and compares full pixel hashes where the
verified public validation corpus is available. Packaged image filename suffixes
hash original **paths**, not pixels, and must not be used as pixel digests.
Historical measure numbers remain candidates, not certified physical-source
measure indexes. The legacy alignment stage sorts and zips scan systems with
symbolic segments; recovery of that provenance does not qualify those joins.

Completed candidate report: `crop-mapping-candidates.json` in that evidence
directory. Train: metadata for 34,510/35,226 windows, archived pixels for
35,222; all 716 missing metadata rows belong to `sq10675759`. Valid: metadata
and archived pixels for all 3,911 windows, with 3,761 exact published pixel
matches and 150 differences across all nine scores. Training public pixels
were not downloaded/compared; the reported zero public train matches means
unmeasured, not 35,226 mismatches.

Metadata internal score/page/system/part checks match only 14,458 train and
1,711 valid rows. This is not automatically evidence of wrong musical labels:
the alignment copies synthetic metadata without rewriting pagination. For
example `sq7383977:0003:0001.yaml` says `page_idx: 1`, `system_idx: 1` and
measures 1–6. Scan page 3 and synthetic page 1 are separate coordinates.
Neither filename equality nor copied page fields certifies physical measure
ownership. Next qualify the historical joins against source windows and scan
evidence, investigate the 150 pixel differences, and keep the 716 unmapped
training windows outside any timing-label replay until recovered.

All recovery jobs finished. One audit regression test verifies full pixel
comparison (including deliberate mismatch) and that candidates never become
certified; the eleven existing page-certification tests also pass. Scoped
lint passes. No corpus, model or runtime pin changed.
