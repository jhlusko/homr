# Lieder rest stretch and actual crop support — 2026-10-04

All 29 previously unresolved duration conflicts are rests whose native staff time-signature stretch explains the discrepancy exactly. They are not evidence of missing dots or low-quality pieces. Fourteen reach the frozen training set, in three crops of Causeries de vagues; the other 15 are outside accepted crop boundaries. Keep elapsed/ratio supervision masked while the source-time representation is resolved; written rest labels remain usable.

## Native source and exporter evidence

The audit locates each event in hash-verified MusicXML, then requires matching native part/staff and physical-measure inventories, rest ordinal, written type and dots. It reads the active native time signature, including hidden changes and resets to stretch 1. A separate native tuplet or mismatched inventory refuses this explanation.

| Score | Conflicts | Native stretch | In indexed training crops |
|---|---:|---|---:|
| IMSLP622105, Dulde, gedulde dich fein | 3 | 3/2 | 0 |
| IMSLP624193, Causeries de vagues | 21 | 2/3 vocal; 3/2 lower piano staff | 14 |
| IMSLP632172, Nourmahal-la-Rousse | 5 | 2/3 | 0 |

For every case, raw XML elapsed quarters × native staff stretch equals the written rest duration. Examples: a vocal eighth rest exported as 3/4 quarter has native stretch 2/3 and written duration 1/2; a lower-staff sixteenth exported as 1/6 has stretch 3/2 and written duration 1/4. These are exact native-source relationships, not rounding or inferred missing markings.

All three frozen MusicXML files identify MuseScore 4.5.2 as their exporter. Its [export source](https://github.com/musescore/MuseScore/blob/v4.5.2/src/importexport/musicxml/internal/musicxml/export/exportmusicxml.cpp) corrects chord-note durations for staff stretch, while rests write actual ticks and their original written type/dots. This asymmetry is consistent with the observed rest-only discrepancy. The pinned source file and its SHA-256 are archived locally. The export diagnosis is an inference from that code and the exact native arithmetic; no exporter rebuild or native-to-XML reproduction ran.

The original scans of all three affected training crops were inspected. They show the expected undotted lower-staff sixteenth rests and undotted vocal eighth/sixteenth rests. Do not manufacture dots or tuplets to reconcile the XML arithmetic. Native staff-specific meters also require care with a shared timeline: changing duration fields alone is not a qualified repair of backups, other staves or event onsets.

The three affected-score replay checks all 60 indexed crops, with token bytes identical and the 14 events traced through the actual parser, staff merger, measure cutter, natural stripping, numerator handling, serialization and sidecar reload. Diagnostic source-event IDs exist only in these audit copies; the public label contract is unchanged.

## Full actual Lieder crop replay

One CPU thread, persistent tmux and completion watcher: all 212 indexed scores / 3,867 crops finish in 358.6 seconds. Crop image hashes and certified source ranges are retained. This reuses actual published crop boundaries; it is not image re-detection or a complete corpus rebuild.

| Split / replay parity | Crops | Timed crop occurrences | Elapsed targets | Written targets | Ratio targets | Rest-flag targets |
|---|---:|---:|---:|---:|---:|---:|
| Train, byte-identical | 3,409 | 128,120 | 128,106 | 126,986 | 126,972 | 10,139 |
| Valid, byte-identical | 291 | 10,607 | 10,607 | 10,544 | 10,544 | 688 |
| Train, changed tokens | 158 | 6,957 | 6,957 | 6,944 | 6,944 | 401 |
| Valid, changed tokens | 9 | 309 | 309 | 301 | 301 | 26 |

The 14 conflict targets remain masked for elapsed/ratio, while their written/rest-identity labels remain available. There are no values outside the frozen mixed-source training classes. The identical training subset supports 23 elapsed classes, 15 written classes, six ratios and both rest flags, covering every value in the identical Lieder validation subset. Many mixed-source classes have no Lieder crop support; the report lists them individually. Do not prune them before measuring OSSQ crop support or add classes from validation.

### The 167 changed crops are a migration gate

Full aggregation corrected the interim claim of no token differences: 3,700 crops are byte-identical and 167 differ. The changed subset is kept separate from candidates for metadata-only migration.

- 47 differ only in serialized group boundaries.
- 95 change note/rest ordering while preserving the identity inventory.
- 25 preserve note/rest ordering but change other token fields or grouping.

None changes the multiset of note/rest rhythm, pitch, accidental and staff-position identities. This does not establish musical equivalence: treating successive grace notes as a chord can lose their order, and grouping changes can affect decoding. Examples include merged grace-note lines, reordered grace pitches, and joined upper/lower chord groups. The cause and correctness need a controlled parser/merger/serializer comparison against the frozen build before these crops are migrated or rebuilt. These are token comparisons, not model scores.

## Review and local artifacts

[Local evidence viewer](http://localhost:8765/) shows native/XML durations beside original training scans, with printed measure and native staff beat labels. All 29 cases are selectable; excluded cases have textual source evidence rather than guessed scan alignment. No individual approval decisions are requested. The server runs in persistent tmux, bound to localhost.

Artifact parent: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`.

- `lieder-rest-stretch-audit-20261004/`: initial three-score audit, all 29 native locations, source hashes, exact exporter source, scans and `review/index.html`.
- `lieder-real-crop-timing-support-20261004/`: full crop replay, per-score records, regenerated audit sidecars, image hashes, `crop-class-support.json`, `legacy-differences.json`, log and completion receipt.

Twenty-eight focused tests pass, including inherited stretch reset, hidden-rest ordinal alignment, native inventory guards and refusal to explain independent tuplets. Earlier codec, sidecar and target/checkpoint tests are included. Audits complete successfully; remaining ETA zero. Only the idle review server remains running.

## Next work

Prioritize the 167 serialization differences, particularly sequential grace notes, before a public corpus update. Recover certified OSSQ crop-to-source boundaries before claiming its actual training support; the inspected local sq8482283 source has no scanned systemwise metadata. Then freeze raw-source and bounded-quantization scoring views with an explicit policy for native staff stretch, and wire opt-in resumable frozen-core head training on qualified labels. Do not apply a blanket rest-duration or ratio correction.

The existing four-head target contract and class manifest are unchanged. No corpus/model/provider pins, runtime behavior or trained-model claims change in this audit. Immediate investigation and trainer wiring can run locally without GPU.
