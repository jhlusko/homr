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
