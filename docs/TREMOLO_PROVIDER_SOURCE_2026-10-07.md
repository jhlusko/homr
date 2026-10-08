# Provider paired-tremolo source candidate — 2026-10-07

Prepared `OurTextScores/services/ots-homr-modal/patches/homr-paired-tremolo-candidate.patch`
as an isolated additive runtime patch. It is not added to the deployed image builder.
It changes only the XML exporter hooks and adds `homr/tremolo_timing.py`, retaining
the rule implementation from HOMR commit138af38 without unrelated newer exporter
changes. No graphs, dependencies or runtime environments were replaced.

## Correct pinned base

Inspection of `modal_app.py` identifies the actual current archive as
`homr-b6eca7-source.tar.gz`, source commit
`b6eca7c6af37df2764a9ca99033c947eb42aae90`, SHA-256
`999d5071fe9bcfab8dc1534ec1614e125888651fcc6c72a9862bc605ba38d1a2`,
followed by `homr-runtime-glue.patch`. The README's older 997b70/two-patch description
is historical and does not describe the current builder. The new rule still requires
an additional runtime patch, even with this newer base.

Candidate patch SHA-256:
`5bbfe5c35e0e6f68526f4e4dd3eb373ae1808e86a53974ac077eb88a06f12da0`.

## Checks

Clean extraction, archive hash validation, existing glue application and candidate
`git apply --check` succeed. Four synthetic exporter probes have byte-identical
flag-off XML compared with the pinned base. Enabled paired-tremolo timing changes
the positive case; ordinary notes, single-note tremolo and changing-meter controls
stay identical. Five runtime guard tests pass against the isolated candidate, as do
26 focused provider/service tests including real HTTP XML regeneration. Scoped
lint passes. These are source compatibility checks, not corpus accuracy or a full
model-loaded scanner test. The isolated imported module path was verified explicitly.

Reproducible verifier:
`/home/jhlusko/workspace/OurTextScores/services/ots-homr-modal/verify_tremolo_source_candidate.py`.
Run with `--python` pointing to an existing HOMR-capable Python and `--output` for
its receipt. It uses a temporary extraction and never loads models or changes the
workspace runtime. Receipt:
`/home/jhlusko/workspace/homr-artifacts/provider-tremolo-source-20261007/source-candidate-report.json`.

Next: a separate candidate service build/configuration that applies this third
source input, carries its hash in provenance and identifies actual graph pins;
then image-first on/off service qualification and full OCR/text-fusion latency.
Keep the existing shipping builder unchanged during that qualification. No deploy,
push, model-pin or backend environment change occurred. All jobs complete.
