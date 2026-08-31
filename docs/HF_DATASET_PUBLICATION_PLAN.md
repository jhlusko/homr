# Hugging Face publication plan: OSSQ and Lieder scan corpora

## Purpose and release boundary

Publish two reproducible, score-disjoint OMR training/evaluation datasets derived from
OpenScore editions and scanned music.  Each release must be usable after download without
the build machine: images are real files (not symlinks), index paths are relative, every
sidecar resolves, and an immutable manifest identifies the exact release tree.

This plan covers the Stage 2 symbolic staff-crop corpora first.  Page-level text/OCR
material is a separate optional configuration, not an accidental payload in v1.

| HF repository | v1 content | source status | proposed visibility |
| --- | --- | --- | --- |
| `ourtextscores/ossq-omr` | scanned and synthetic OSSQ staff crops; tokens; notation sidecars; score-level split manifests; provenance and audits | OpenScore String Quartets + OSSQ scans; CC0 stated by the current distribution record | public after provenance check |
| `ourtextscores/lieder-omr-private-all-sources` | complete internal 215-source, 3,922-row v4 boundary-safe corpus; source ledger and rights statuses | OpenScore Lieder symbols are CC0; images are individual IMSLP uploads | **private**; full provenance/reconstruction store, never an open-release claim |
| `ourtextscores/lieder-omr` | only the cleared public candidate subset: 25 sources / 565 rows, tokens, notation sidecars, score-level split manifests and provenance | exact IMSLP file records marked `Public Domain` + `Normal Scan`; OpenScore symbols CC0 | public candidate only after final human rights check |

Do not publish model weights, held-out prediction outputs, reviewer corrections, raw
unreviewed alignments, or the historical recovered/model-derived Lieder labels as dataset
examples.  They belong in separate model, benchmark, or research-artifact repositories.

## Dataset contract

Each row is one staff crop and has enough information to train HOMR's Stage 2 transformer
without project-local paths.

```text
splits/train/index.txt       image.png,tokens.tokens
splits/validation/index.txt  image.png,tokens.tokens
notation/<id>.notation.json  structured labels aligned to the token sequence
```

The exact final layout can differ, but these invariants may not:

1. Splits are by source score, never random crop.  Release a `score_id` manifest and a
   machine-checkable assertion that no score occurs in more than one split.
2. Every token stream must round-trip through the pinned parser/render checks chosen for
   that release.  Report failures; do not silently delete them after split creation.
3. Sidecars are optional at read time but, when present, must share the crop ID and token
   length.  Document their schema and label vocabularies (beam levels, ties, stem,
   slur, advance, dynamic).
4. The release includes a `MANIFEST.json` with file count, byte count, and tree digest,
   generated after all paths have been rewritten.  The manifest excludes itself.
5. No file, index row, JSON image reference, or symlink may point outside the repository.

## Lieder-specific admissibility rule

Only the v4 boundary-safe, model-free consensus corpus may be considered for the Lieder
release.  A released row needs an explicit certified crop-to-reference measure range.
Rows marked `boundary_ambiguous`, historical ordinal alignments, and anything recovered
from a recognizer match are excluded from all public train and validation splits.

The dataset card must explain the central limitation plainly: MuseScore line wrapping can
make one scanned line correspond to multiple reference lines.  The release therefore
does not infer individual cuts from a matching group total.  Link the shipped alignment
ledger and [`LIEDER_ALIGNMENT_REBUILD.md`](../training/omr_datasets/LIEDER_ALIGNMENT_REBUILD.md).

## Rights, licensing, and provenance gates

### OSSQ

Before upload, create a machine-readable work table containing source work identifier,
OpenScore source URL/revision, scan source URL, licence assertion, and every released
crop ID.  Verify the current CC0 claim for both the OpenScore transcription and the OSSQ
scans.  Put the resulting licence identifier in the dataset card and `README` metadata.

### Lieder

Do **not** select a blanket HF licence for the scans from the fact that the OpenScore
transcriptions are CC0.  IMSLP scan terms are uploader-specific.  For every source score,
produce a provenance table with IMSLP work/page IDs, source URL, uploader/licence status,
retrieval date, and the released crop IDs derived from it.

Choose one of these routes before any public upload:

1. obtain and record a compatible release basis for every included scan;
2. publish a rights-cleared subset with its own score-disjoint split and clearly state the
   coverage change; or
3. keep images gated/private and publish only code, manifests, and instructions for a
   user-authorised source reconstruction.

The card must not describe a gated or private repository as an open dataset.  Legal or
rights review is a release blocker, not a documentation task to defer.

### 2026-08-31 IMSLP exact-file audit

The release decision now has an auditable source-level basis. Every one of the **215
IMSLP file IDs** represented in the v4 boundary-safe indexes (**3,922 crop rows**) was
resolved through the official IMSLP MediaWiki record. A work title or composer lifetime
was never used as a proxy for its downloaded scan: the audit binds each local
`IMSLP<n>` source to the exact file record and captures filename, uploader, submission
date, publisher/edition information, stated copyright, image type, and stable
`Special:ImagefromIndex` URL.

| audit status | source files | crop rows | release decision |
| --- | ---: | ---: | --- |
| `candidate_pd_scan` | 25 | 565 | eligible only for the conservative public-candidate route below |
| `exclude` | 64 | 992 | not an original `Normal Scan`; never include in a public image release |
| `needs_manual_review` | 126 | 2,365 | missing/ambiguous automatic file binding; private only until manually resolved |

`candidate_pd_scan` means the exact IMSLP record says both `Copyright=Public Domain`
and `Image Type=Normal Scan`, and identifies an edition/publisher. It is a documented
release candidate, **not** a blanket CC0 assertion and not a substitute for a final
human/legal review in the intended release jurisdictions. The full ledger is
`lieder-rebuild/imslp_rights_audit_v4.{csv,jsonl}`; the frozen allow-list is
`lieder-rebuild/imslp_release_candidate_v4_score_ids.txt`; filtered source-disjoint
indexes contain 499 train rows from 22 scores and 66 validation rows from 3 scores.

The public candidate card must preserve individual source links and say that
OpenScore-derived symbolic targets are CC0 while the scan-image basis is the exact
IMSLP record's public-domain assertion. Use HF's `other` licence field and this explicit
rights statement unless final review establishes that `public-domain` accurately
describes the combined package. Do not call the combined image/token repository CC0.

The private all-source repository is intentionally broader: it stores the 215-source
corpus and complete audit ledger for approved collaborators, including excluded and
unresolved records. Private hosting does not resolve rights; it is access control and
provenance preservation only. Do not transfer it to a public/gated repository by a
repository-visibility toggle. A public release must be rebuilt from the 25-ID allow-list
and independently verified.

## Build and validation procedure

1. Freeze source revisions and write `release.json` with repository commit, conversion
   command, parser/vocabulary version, source-root IDs, selected score IDs, and date.
2. Build into a new staging directory; never package the training directory in place.
3. Run `training.omr_datasets.package_dataset` on each staged root.  It materialises
   symlinks, rewrites absolute index/page paths, creates `MANIFEST.json`, and verifies the
   result from disk.  Use `--verify-only` again after the upload/download smoke test.
4. Run corpus checks: index readability, split disjointness, token/sidecar alignment,
   duplicate image/token detection, token parse/render round trip, and release-specific
   Lieder provenance/alignment audit.
5. Make a clean-directory consumer test: download one split, run the public loader with
   only relative paths, train one small batch, and reproduce a documented evaluator
   invocation.  This catches the symlink and build-path failures that ordinary file
   counts miss.
6. Create deterministic archives only if HF file layout/size requires them.  Prefer
   browsable Parquet/JSONL metadata plus image files where practical; use LFS for large
   binary assets and document exact extraction commands for any archives.
7. Upload a release candidate to a private HF repository, clone it fresh, repeat step 5,
   compare `MANIFEST.json`, then tag the immutable release (`v1.0.0`).

## Repository layout on Hugging Face

```text
README.md                    dataset card and YAML metadata
LICENSE                      only after the rights gate is satisfied
release.json                 build provenance and software versions
MANIFEST.json                tree digest, count, bytes
splits/
  train/index.txt
  validation/index.txt
  train-score-ids.json
  validation-score-ids.json
data/                        crops, tokens, notation sidecars
provenance/
  works.jsonl                source work/page mapping and terms
  alignment-ledger.jsonl     Lieder only; released-row measure ranges
docs/
  SCHEMA.md
  REPRODUCE.md
  QUALITY.md
  EXCLUSIONS.md
```

Keep image bytes, tokens, and sidecars together by stable crop ID; do not require consumers
to infer a filename from an absolute source path.  If metadata is also published as
Parquet, treat it as an index/view over these canonical files and test that both agree.

## Dataset-card minimum content

- intended use: staff-level OMR research and training, not a benchmark of general music;
- composition: works, pages/crops, staff types, scan/synthetic balance, split counts, and
  repertoire limits (OSSQ is string-quartet-centric; Lieder is voice plus piano);
- source provenance and licences, including the Lieder gate status;
- preprocessing: crop generation, MusicXML-to-token conversion, clef handling, sidecars,
  and image normalisation;
- split strategy and score-disjointness proof;
- quality controls and known exclusions;
- known limitations: engraving-specific line breaks, rare notation classes, transcription
  inaccuracies, OCR/text labels if offered, and non-comparability of absolute accuracy
  across OSSQ and Lieder;
- reproduction commands, manifest verification, citation, and contact/removal process.

Do not lead with a model score.  Report dataset facts and audits separately from any model
trained on it, and label all validation subsets as development splits rather than an
independent external benchmark.

## Publication checklist and owners

| Gate | Evidence required | OSSQ | Lieder |
| --- | --- | --- | --- |
| Rights | per-work provenance table and licence decision | required | required; blocks public release |
| Data selection | frozen score IDs and exclusion ledger | required | v4 boundary-safe only |
| Portability | packager + verify-only clean-download pass | required | required |
| Integrity | manifest digest and split/sidecar/round-trip reports | required | required |
| Card | reviewed README, schema, limitations, citation | required | required |
| HF release | private candidate cloned and tested; version tag | required | required |

## Proposed sequence

1. Publish an OSSQ private candidate first; it has the clearer CC0 path and exercises the
   packaging/upload/re-download procedure.
2. Turn the candidate's generated provenance, manifest, and clean-consumer test into a
   reusable release script/template rather than hand-maintained instructions.
3. Create the private all-source Lieder repository with the complete 215-source ledger,
   frozen v4 provenance, and access-controlled artifacts. It is the reproducible
   collaborator store, not a public-release candidate.
4. Build the public Lieder candidate only from the 25-ID public-domain normal-scan
   allow-list, rerun score-disjoint split and portable-package validation, and perform
   final human review of all 25 linked IMSLP records before any public upload.
5. Announce immutable version tags, not a moving `main` branch.  Corrected future rows
   become a new version with a changelog and preserved prior manifest.
