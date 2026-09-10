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

## What a finding is, and who sees it

A profile is a prior supplied by a human who may be wrong, so the question "what
happens when the profile is wrong?" is the design, not an edge case. A profile expecting
the cello in bass clef meets a movement that genuinely opens in tenor, or a passage that
goes into treble; both are ordinary music, not recognition errors.

**Today the answer is bad: findings go to `eprint`.** `check_clefs_against_profile`
emits a `Finding` that `_report_cross_staff_findings` writes to the process log, which
in production is a Modal container's stdout. No reviewer will ever see it. Shipping the
channel without a surface would mean a wrong profile silently produces log noise, and a
*right* profile catching a real error also silently produces log noise. Neither is worth
building.

So the channel is only worth landing together with a surface, and the surface must be
built for the case where the **profile** is wrong rather than the score:

1. **Findings are page-scoped review annotations, not errors.** A clef finding says
   "staff 3 decoded treble; the profile expects bass or tenor for Cello" and offers two
   dispositions: *the reading is wrong* (a recognition problem, route to the existing
   review queue) and *the profile is wrong* (amend the profile for this job).
2. **Amending the profile is the expected outcome, not a failure path.** A cello part
   that uses tenor clef should end with `likelyClefs: ["bass", "tenor"]` recorded against
   that job, and the finding should not recur on later pages. This is why `stableId` is
   scoped to the submitted job rather than a universal instrument registry — the schema
   already anticipated exactly this.
3. **A dismissed finding stays dismissed, per part and per clef.** A quartet with a
   tenor-clef cello would otherwise raise the same finding on every page of the
   movement, which trains reviewers to ignore the whole category.
4. **Findings never gate anything.** They do not block assembly, mark a page failed, or
   change `page.status`. A page with unresolved profile findings is a complete page with
   review annotations on it.
5. **Provenance records which side won.** When a reviewer amends the profile, record it,
   because a profile amended three times in one score is evidence that the profile was
   guessed rather than known — worth knowing before anyone proposes giving profiles
   corrective authority.

The precision consequence is worth stating plainly: clef findings will be **noisy on
real repertoire**. Cello tenor/treble, viola treble, bassoon tenor and octave-transposing
voice parts are all common. A finding that fires on ordinary music is only tolerable
because it is an annotation a reviewer can dismiss in one click, and it would be
intolerable if it were an error, a block, or an automatic correction. That asymmetry is
the whole reason §"Principle" refuses corrective authority.

Surface seam: the findings ride the existing scan response as a bounded per-page
artifact alongside the Stage 3 diagnostics, and render in the scanner page card next to
the engine list — the same place a reviewer already looks at per-page state.

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
- A clef finding survives to the response as a review annotation, and a dismissal
  suppresses it for that part and clef on later pages of the same job.
- Fingerprint: two runs differing only in profile do not share an idempotency key.
