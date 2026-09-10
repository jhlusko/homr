# Score profile channel

**Status:** proposed design
**Date:** 2026-09-10
**Scope:** a path for a caller-supplied `ScoreProfile` to reach `homr.main.ProcessingConfig`,
so the clef and layout checks that already exist stop being unreachable.

## Problem

`homr.score_profile.ScoreProfile` is built, schema-versioned, dependency-light, and
explicitly designed to be serialized by a caller outside the package. Two consumers
exist: `check_clefs_against_profile` (a decoded clef the profile did not expect) and
`propose_part_assignment`'s layout deviations (a system's staff count disagreeing with
the profile).

Neither can ever run in production. The service builds `ProcessingConfig` with nine
positional arguments and leaves `score_profile=None`, and the provider's whole option
surface is one `textDetector` string, validated by `Object.keys(value).length === 1`.
There is no channel, so there is nothing to supply.

This is the cheapest large win available for ensemble scores. For a string quartet the
parts, their order, and their clefs are known before recognition starts.

## Principle

The profile is **a prior a reviewer can see and reject, not an assertion the pipeline
enforces.** `score_profile.py` says this in its own module docstring and the design must
not quietly promote it. Concretely: a profile may raise findings and proposals; it may
never override a decode on its own authority, and a wrong profile must degrade to
today's behaviour rather than corrupting a page.

That rules out the tempting version of this feature — "the profile says 4/4, so write
4/4" — and it is the right call. A profile is supplied by a human who may be wrong about
which edition they uploaded.

## Contract

Extend the engine option as a discriminated union, the same shape the text-fusion design
uses, rather than overloading the existing scalar:

```ts
type OtsHomrEngineOptions = {
  textDetector: 'lyrics' | 'non-lyric-text' | 'none';
  scoreProfile?: {
    schemaVersion: 'homr.score-profile.v1';
    profileSha256: string;   // over the canonical JSON, for the fingerprint
    parts: Array<{
      stableId: string;
      displayName?: string;
      likelyClefs?: string[];
      transposition?: number;
    }>;
  };
};
```

`isScannerOtsHomrEngineOptions` currently asserts exactly one key. That check becomes an
allowlist of known keys — it exists to reject unknown policy, and widening it to two
named optional fields keeps that property.

**The profile joins the idempotency fingerprint.** `preprocessingRevision` and
`decodingRevision` already carry detector and heads identity; add `profileSha256`. A
profile change must create a new attempt, never relabel an existing result — the same
rule the text-fusion design states, and for the same reason.

## Seams, in order

1. **`backend/src/scanner/scanner-dual-engine.ts`** — widen `ScannerOtsHomrEngineOptions`
   and its validator. Profiles are frozen into the engine plan alongside `textDetector`,
   so a mid-job edit cannot retroactively change what a completed page was read with.
2. **`scanner-ots-homr-provider.service.ts`** — add `profileSha256` to the idempotency
   key; `form.set('scoreProfile', JSON.stringify(profile))` on the request;
   `verifySelectedComponents` asserts the response echoes the same SHA.
3. **`services/ots-homr-modal/ots_homr_provider.py`** — parse and validate against
   `homr.score_profile.parse_score_profile`, which already raises
   `ScoreProfileSchemaError` on a version mismatch. A malformed profile is a 4xx refusal,
   not a silent drop: silently ignoring it would make the feature untestable from
   outside.
4. **`services/ots-homr-modal/homr_engine.py`** — the positional `ProcessingConfig`
   construction becomes keyword arguments (nine positional booleans is how
   `score_profile` came to be omitted in the first place) and passes `score_profile=`.
5. **`homr`** — nothing to change. `parse_staffs` already threads it to
   `_report_cross_staff_findings`.

## Where the profile comes from

Deliberately out of scope for the channel, and the reason to build the channel first:
once it exists, any of these can fill it without further plumbing.

- **Reviewer-supplied**, from the scanner page-setup UI: the strongest signal and the
  smallest build. "This is a string quartet" is four parts and their clefs.
- **Derived from an existing work** in OurTextScores when a scan is uploaded against a
  known work.
- **Inferred from page 1** — detected staff count, brace/bracket structure, and any
  part names OCR read. This is the most valuable and the most dangerous: an inferred
  profile applied to its own page is circular, so it must be inferred from one page and
  applied to the *others*, and marked as inferred in provenance.

## What it unlocks, in order of value

1. **Clef findings** (`check_clefs_against_profile`) — works the day the channel lands.
   A cello staff decoded with a treble clef against a profile expecting bass is a
   high-precision finding.
2. **Layout deviations** (`propose_part_assignment`) — a system whose detected staff
   count disagrees with the profile is usually a detection failure, and today it is
   invisible.
3. **Part identity in output** — `stableId` gives merged/compare surfaces a stable part
   key across engines, which the OurTextScores compare view currently derives
   heuristically.
4. **A metre prior, as a finding only.** With §4.2's tier-1 repair now applying majority
   corrections, a profile could break a 2–2 tie that the majority rule correctly refuses
   to guess at. This is the one place a profile could plausibly earn corrective
   authority, and it should be a separate, benchmarked decision — not folded into the
   channel.

## Testing

- Round-trip: a profile through the Scanner option, the provider form, the service
  parse, and back out in provenance, unchanged and SHA-identical.
- A version mismatch refuses rather than degrades.
- Absent profile is byte-identical to today's output — the regression assertion that
  matters, and the same shape the text-fusion design uses for core dynamics.
- A deliberately wrong profile changes findings only, never decoded symbols.
- Fingerprint: two runs differing only in profile do not share an idempotency key.
