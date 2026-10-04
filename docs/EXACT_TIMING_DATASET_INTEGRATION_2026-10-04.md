# Exact timing dataset integration — 2026-10-04

Exact reference timing now survives the current HOMR MusicXML parser, staff merger, crop selection, token serialization and sidecar attachment. The token vocabulary, token durations and legacy model targets are unchanged. Model losses and runtime export do not consume these labels yet.

## Versioned contract

Timed non-grace pitched notes and rests carry `EncodedSymbol.exact_timing`, outside equality, hashing, string serialization and decoded duration. Grace/control symbols remain unannotated. The metadata records exact rational elapsed quarters, written type/dots, full-measure-rest identity, time-modification XML, and nested tuplet elements with their source notation-container index and attributes (including visibility). The label schema is `homr.exact-timing-label.v2`; the isolated audit adapter still understands the preceding prototype-v1 payload.

The shared notation writer also writes `<tokens>.timing.json`, with file schema `homr.exact-timing-sidecar.v1`. Records follow the same `serialized_chords` order as the token writer, including explicit null timing entries for grace/other note-bearing symbols. A SHA-256 of the token file and each symbol's rhythm/pitch/lift/position guard attachment. Same-count token edits and wrong symbol order are refused before mutation. A rebuild with no timing metadata removes an obsolete timing sidecar.

The existing `.notation.json` schema and return/count semantics remain unchanged. Existing token/notation pairs with no timing file load as before. Current training loaders that call `attach_sidecar` receive reference metadata, but losses ignore it. This preserves checkpoint compatibility rather than pretending a reference sidecar supplies model predictions.

## Implementation and bounded checks

Parser extraction attaches metadata before grouping; the existing merger and measure selection keep the same symbol objects. Copying and token-field transformations retain the metadata while legacy textual identity stays unchanged. Both current HOMR OSSQ and Lieder converters use the shared writer after selecting crop measures.

Forty-eight focused tests and three subtests pass. Coverage includes crop selection, full-rest identity, exact duplet duration, grace placeholders, serialization ordering, notation-container visibility/nesting, changed-token hash rejection, wrong-order rejection, stale-file removal and old sidecar compatibility. The three-score real preflight (IMSLP598858, IMSLP622488, IMSLP89026) exactly replays preceding advance-off/on XML hashes, metrics and timing; 18 selected windows / 499 timed symbols roundtrip without unsupported legacy crop tokens.

## Full qualification

Complete: all 297 indexed references checked in 733.0 seconds (12.2 minutes), one CPU thread, persistent tmux/watcher EXIT=0. Legacy XML hashes, metrics and timing replay exactly with advance off/on for 75/85 OSSQ and 212/212 Lieder scores. All ten OSSQ detachedLegato export failures reproduce the old stage/message in both arms. No new control mismatch or metadata alignment failure occurs.

| Source | Selected label windows verified | Timed symbols retained | Unsupported legacy windows |
|---|---:|---:|---:|
| OSSQ | 1,235 | 16,381 | 2 |
| Lieder | 1,137 | 27,426 | 0 |

The two OSSQ gaps are sq12772795, merged part 1 measure index 214 (`note_72`), and sq8928855, merged part 1 measure index 20 (`note_19`). These are selected canonical windows that the existing vocabulary cannot serialize; no metadata success is claimed for them. Source-index locations are not interchangeable with original XML printed measure/staff labels. The additive integration qualifies for the tested serializable windows, with these explicit coverage gaps. It does not expand the vocabulary or make all canonical windows trainable.

Selected leading/trailing, full-rest and tuplet windows pass token serialization/reloading and exact sidecar comparison, including on references with known export failures. Windows are deterministic contiguous measure selections, not actual image-boundary crops or a full corpus rebuild. The updated v2 contract also passes selected-field reconstruction for all 2,565 located canonical problem events (2,387 OSSQ, 178 Lieder), including notation-container ownership/visibility, in 9.1 seconds. All jobs finish EXIT=0; remaining ETA zero. This is not a model decoding gate or a recognition improvement.

Local parent: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`.

- `exact-timing-pipeline-preflight-20261004/`: three-score report, selected label files and source/control hashes.
- `exact-timing-pipeline-qualification-20261004-v2/`: full outcomes, selected real label windows, launch/log/EXIT/watcher and tested source snapshots. The initial unsuffixed run retains an audit-driver failure on a previously known detachedLegato export; v2 requires that old failure to replay exactly rather than treating it as a new regression.
- `exact-timing-representation-20261004-v4/`: all located problem events rechecked with the v2 ownership/visibility contract, source/reference hashes and completion records.

## Corpus and model follow-up

The public OSSQ/Lieder repositories deliberately build against immutable `vendor/homr-stage2` snapshots. Those snapshots and public corpus versions have not been changed. A future corpus release must explicitly vendor the qualified HOMR commit, include `.timing.json` in packaging, and regenerate labels/crops with receipts; otherwise the old builders will not gain this metadata.

Next, define model-predicted exact-duration and written-notation targets plus explicit checkpoint migration. Reference timing must never be substituted into production decoding or used to claim recognition improvement. Then qualify new model decoding on OSSQ/Lieder and package versioned corpora. The additive labels do not resolve legacy duplet rounding in exported/model-predicted tokens; they retain the missing information for that migration. Predicted advance remains disabled. No GPU is needed until model training begins.

See `EXACT_TIMING_REPRESENTATION_2026-10-04.md` for the canonical census and limitations of the earlier isolated prototype.
