# Isolated tie-alias XML experiment — 2026-10-05

The saved eight page/model outputs were replayed with a source-free, confidence-gated XML experiment. It considered only complete phrase spans that exactly duplicated both exported sounding and engraved ties. At both endpoints, both phrase slots had to be `none`, and the tie state had to own the matching start or stop with probability at least 0.90. No runtime code, default, model or saved output was changed.

The experiment removed 3 candidate spans on Franz, 15 on OSSQ page `sq7358579`, 3 on OSSQ page `sq9146376`, and none on the remaining arms. Every arm passed invariants: ties were byte-for-byte unchanged, non-slur XML was unchanged, and only the selected phrase endpoints were removed. This is a structural safety result, not evidence that the removed spans are visually wrong.

The test is intentionally conservative. It cannot repair unspecified sides, infer phrase ownership, or distinguish a true phrase beginning on a tied note without complete endpoint evidence. It also does not measure full recognition, rendering, or scanned engraving quality. The one known source-correct Franz phrase duplicated by a predicted tie remains protected by the confidence/event conditions.

Next: render the isolated outputs for visual inspection, add independent true-phrase and boundary controls, and compare the complete stack on the same page metrics before considering an opt-in policy. Keep the default exporter unchanged.

Local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/alias-xml-experiment-20261005/`.
