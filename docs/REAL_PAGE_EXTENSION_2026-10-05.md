# Additional real-page certification and comparison — 2026-10-05

Four additional real music pages have hash-bound agent visual edition, part-order and complete-page/system boundary checks. Selection was frozen before recognition. This is diagnostic coverage, not an untouched test or promotion.

| Source / score | Scan page | Physical reference window (zero-based/end excluded) | System counts |
|---|---|---|---|
| OSSQ Haydn Hob.III24 Op.9 No.6, sq9146376 | 0002 | [0,21) | 3,5,4,5,4 |
| OSSQ Glazunov Op.1, sq7358579 | 0002 | [0,18) | 4,4,5,5 |
| Lieder Chaminade Ballade à la lune, IMSLP154086 | p004 | [33,44) | 3,3,2,3 |
| Lieder Franz Sonntag, IMSLP89797 | p002 | [19,44) | 5,5,5,5,5 |

Haydn includes native measure 0 (pickup): native numbers 0–20. Franz also has an earlier measure 0 pickup: this page has native 19–43. Chaminade native 34–44; Glazunov native 1–18. Physical indices and printed numbers are not interchangeable.

## Selection and evidence

OSSQ uses the next two identities in the previously frozen 17-score pool. Both first scanned images are covers. CSV's page2 hint is independently verified visually against canonical source previews and distinct musical anchors for every system. All four source parts agree on boundaries/native numbers. This is edition/boundary review, not exhaustive note-level fidelity certification.

Lieder uses independent `alignment_v4_boundary_safe.json` and the existing inventory. Fixed 10 MP budget; require local scans, unique independent source identity, complete contiguous systems and consistent native part counts. Exclude recovered public-core/joint training-index identities and already-compared Lorelei. Scores sort by SHA256(`full-page-extension-20261005:` + identity); one page per score sorts by hashed filename. Exactly two additional scores remain. Both retain known validation/historical-index exposure; OSSQ scanned-test scores were previously evaluated. Historical old-model/head training provenance remains incomplete. No untouched-test claim.

MuseScore 4.6.5 renders display-only windows from unchanged native XML/MXL. Preview slicing carries prior staff-specific clef/key/time/divisions. Display reflow spans two preview pages for Chaminade and three for Franz; the scan windows remain one page each. Scans/native identities are authoritative; previews are never regenerated ground truth.

Evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/real-page-selection-20261005/`.

`selection.json`, `selection-receipt.json`, `visual-review.json`, original/native preview images, `certified/pages.json`, `certified/receipt.json` and `executed-source/` preserve hashes. Reproduce with `training/onnx/prepare_page_certification_extension.py` then `certify_page_extension.py`; outputs cannot overwrite an existing selection/certification.

## Paired protocol

Eight sequential decodes use one CPU thread, CPU 7 and reduced priority in persistent tmux `page_extension_20261005`, with `page_extension_watch_20261005`. Accepted September18-core/September24-v7-head and public-joint-core-heads-20260928 epoch27 use previously qualified float32 graphs. Input/model/code identities include parser, candidate reranker, vocabulary cleanup and event scorer. Resume rejects changed identities; failures remain failed without zero-error metrics.

Policies match earlier diagnostics: reranking and cross-staff/tuplet/beam/tie/slur-crossing repairs run; advance/stem-arbitration/slur-side are off; text/title OCR do not run. This is not final D1/provider/fusion-policy or production-latency qualification. V7's 0.8 OSSQ arbiter calibration is not silently transferred to epoch27 heads.

Ordered fixed-staff event scores retain pitch, fractional duration and accidentals. Separate `audit_page_timing_context.py` adds strict page-relative physical-measure/onset and inherited XML-meter edits, without oracle rebar/alignment. Missing/extra bars count as errors; unknown meter remains unknown and mid-bar meter changes are rejected. XML meter may be exporter-inferred, not predicted. These are not crop SER and cannot directly implement the +0.25 pp Lieder token gate. Independent voice-ownership/notation scoring remains open.

Run: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/real-page-pair-20261005/`.

`progress.json` records measured ETA; `completed-arms.json`, `summary.json`, `completion-report.md`, `EXIT`, `WATCH_FINISHED`, source snapshots and per-score `comparison.json`, `edit-audit.json`, `timing-context-audit.json` retain results. Initial ETA 10–15 minutes, revised from measured arms. Text fusion is independent: `homr/docs/TEXT_FUSION_QUALIFICATION_2026-10-05.md`.

No inference defaults, pins or released labels change. No training, GPU rental, upload, push or deployment.

## Completed results

All 8/8 arms succeeded, four fully paired pages, EXIT 0 and WATCH_FINISHED 0. Wall time 768.5 seconds (12m49s). Compute and watchers complete; ETA 0. The review server remains active.

Counts below are accepted → epoch27 edits against the same native reference; the denominator is identical for both arms.

| Page | Pitch | Rhythm | Accidentals | Combined events |
|---|---:|---:|---:|---:|
| sq9146376 | 4 → 4 / 252 | 5 → 3 / 320 | 3 → 4 / 252 | 8 → 6 / 320 |
| sq7358579 | 1 → 1 / 300 | 1 → 1 / 364 | 0 → 0 / 300 | 2 → 2 / 364 |
| IMSLP154086 | 1 → 1 / 247 | 1 → 1 / 254 | 1 → 1 / 247 | 1 → 1 / 254 |
| IMSLP89797 | 6 → 7 / 439 | 8 → 9 / 486 | 6 → 7 / 439 | 8 → 9 / 486 |

Haydn improves rhythm but adds one accidental substitution; Glazunov and Chaminade tie. Franz is one event/pitch/rhythm error worse. Its pitch and accidental projection errors are deletions (coverage), not accidental substitutions. Four selected pages do not establish significance or promotion.

| Page | Strict measure/onset events | Meter-by-measure edits | Physical measures per staff |
|---|---:|---:|---:|
| sq9146376 | 11 → 6 / 320 | 0 → 0 / 84 | 21 |
| sq7358579 | 2 → 2 / 364 | 0 → 0 / 72 | 18 |
| IMSLP154086 | 1 → 1 / 254 | 0 → 0 / 33 | 11 |
| IMSLP89797 | 26 → 27 / 486 | 51 → 42 / 75 | 25 |

All predicted physical measure counts match the references. Franz exposes a shared inherited-meter issue: source meter is 6/4 at page entry, changes to 4/4 at native measure 33, and returns to 6/4 at measure 41. Neither model exports correct initial inherited 6/4. Accepted vocal meter is unknown for 14 measures, accompaniment is 4/4, and final accompaniment incorrectly becomes 3/4. Epoch27 exports 4/4 initially but correctly returns all staves to 6/4 at measure 41. Those are XML-level observations; exporter inference and model-emitted meter are not conflated. No source meter is injected into inference.

The new timing audit explicitly retains per-staff XML meter streams. Input/worker code snapshots bind the executed run; subsequent diagnostic tools are in `postscore-source/`. Later formatting of the supervisor/scorer does not rewrite original receipts. Reproducing the original resumed run requires its saved code identity, not a silently changed checkout.

Next prioritize a predicted, previous-page context experiment with meter-change controls and strict onset scoring, alongside certified final-policy stack qualification. Keep the accepted rollback stack and candidate status; do not enable a cleanup guard or claim promotion from these pages. Broader untouched-score provenance and independent voice/notation evaluation remain open.

Verification: 18 focused page/reference/timing tests pass; Black/Ruff checks pass for the new/modified tooling. No new defaults, flags, pins, training or deployment.

The execution includes pre-existing dirty runtime changes preserved in the run snapshot. This tooling commit alone is not a clean-checkout reproduction of the tested stack; the recorded runtime snapshot and graph identities are required.
