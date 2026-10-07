# Paired-tremolo scanner service wiring — 2026-10-07

The dedicated OTS-HOMR service now carries an explicit default-off paired-tremolo
policy through both core workers, initial image exports and review regeneration.
Set `OTS_HOMR_PAIRED_TREMOLO_TIMING=1` in the provider process configuration to
request it. The worker passes an explicit bool to `XmlGeneratorArguments`, so an
ambient `HOMR_PAIRED_TREMOLO_TIMING` cannot accidentally enable a disabled service.

When enabled, scanner identity and capabilities report
`notationPolicies.pairedTremoloTiming = peer-span-v1`. Disabled configurations omit
that field and retain the old response shape. The worker validates constructor
support before loading weights: an old runtime remains compatible when disabled,
but enabling the rule against it fails readiness rather than silently ignoring it.

## Verification and limits

26 focused provider, engine and core tests pass. These include real HOMR MusicXML
regeneration through the existing authenticated HTTP endpoint, enabled and disabled,
retaining written half-note types while enabling 2:1 consumed timing; propagation
into both worker configurations; policy identity/capabilities; ambient environment
override; and unsupported-runtime rejection. Scoped Ruff and Black checks pass.

The HTTP smoke uses FastAPI TestClient and real XML generation, not live image
inference or model loading. The separate original-image OSSQ/Lieder smoke remains
recorded in `TREMOLO_FRESH_SMOKE_2026-10-07.md`. Neither qualifies full provider
latency with OCR and text fusion. No local full scanner container was launched.

## Remaining release work

The Modal image still builds the older archived HOMR runtime, which does not contain
this rule. Before enabling the service flag, prepare and validate a candidate runtime
source patch/archive and provider source revision. Keep the deployed archive and
model pins unchanged until that candidate is qualified. This is a runtime-source
change; the timing rule requires no new model weights. Current published provider
graphs differ from the accepted experimental graphs used by the fresh smoke, so
that result must not be presented as qualification of the published provider stack.

Next: prepare the isolated provider runtime-source candidate, test its compatibility
with the pinned base, then run image-first service on/off qualification with explicit
graph identities and measure the full path. C1 lyric-versus-direction routing and
E3 final examples remain separate release items. No deployment, push, production
configuration or backend `.env` update occurred. All checks are complete; no job is
running or waiting on a watcher.

Implementation and tests: `/home/jhlusko/workspace/OurTextScores/services/ots-homr-modal/`.
