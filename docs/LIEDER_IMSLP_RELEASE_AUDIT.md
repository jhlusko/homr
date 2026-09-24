# Lieder IMSLP scan release audit

**Status: resolved 2026-09-24 — 212 of 215 sources (3,867 of 3,922 crop rows) are cleared
for distribution.** The original audit, below, cleared only 25; its two defects and the
owner's decisions are recorded in the two sections that follow. This audit checked every
one of the 215 IMSLP file IDs used by the v4 boundary-safe corpus on 2026-08-31.

IMSLP identifiers name individual uploaded files, not just musical works. A work
often has several editions, so composer death dates and a work-level public-domain
statement are not enough evidence for the scan actually used in training. The audit
therefore resolved each local `IMSLP<n>` filename to its rendered IMSLP file block
and retained a source only when that exact block says both:

1. `Copyright=Public Domain`; and
2. `Image Type=Normal Scan`.

It also requires publisher/edition information. This creates a conservative,
traceable candidate; it is not legal advice or permission to claim that all IMSLP
downloads are CC0. IMSLP itself notes that public-domain treatment can be
jurisdiction- and edition-specific, and that a file's terms need to be checked at
the individual record. See [IMSLP’s public-domain guidance](https://imslp.org/wiki/Public_domain)
and [permissible-license policy](https://imslp.org/wiki/IMSLP:Permissible_licenses).

## Resolved, 2026-09-24: 212 of 215 sources cleared

The owner reviewed every source and the position is now settled. The machine-readable
ledger is [`lieder-rebuild/rights_decisions_v4.json`](../lieder-rebuild/rights_decisions_v4.json),
which records, per source, the copyright line its IMSLP record gave, where that line was
read from, and the basis for including it.

| | sources | crop rows |
| --- | ---: | ---: |
| Cleared for distribution | 212 | 3,867 |
| Excluded | 3 | 55 |
| **Corpus** | **215** | **3,922** |

Every cleared source's record states `Copyright: Public Domain`. The 126 that the original
audit could not resolve were resolved by `Special:ReverseLookup/<file id>`: 124 returned a
work page, 59 carrying the copyright line in the file's own block and 65 in the block of
the file set containing it. That extraction was checked against ten sources whose values
the audit had already recorded, including two read through the set-level route, and agreed
on all ten with no disagreements.

`Image Type` remains unread for those 124: IMSLP does not render it on work pages, and it
is available only in the wikitext, which the site serves behind a bot check. The owner
ruled that an absent or unreadable image type does not disqualify a source — consistent
with the correction above, where 61 of 63 sources name the scanner.

The three exclusions, each on its own evidence rather than a missing field:

| IMSLP | Rows | Why |
| --- | ---: | --- |
| 257340 | 28 | Record states `Creative Commons Attribution-ShareAlike 4.0` and `Image Type=Typeset`. |
| 16883 | 22 | IMSLP holds no record for this file id (Satie, *Je te veux*); likely deleted or renumbered. |
| 16400 | 5 | IMSLP holds no record for this file id (Schubert, *Ellens Gesang III*). |

This is a provenance decision by the corpus owner, not legal advice, and it does not
change IMSLP's own caution that public-domain treatment can be jurisdiction- and
edition-specific.

## Correction, 2026-09-24: 63 of the 64 exclusions were wrong

The audit retained a source only when its IMSLP record stated both `Copyright=Public
Domain` and `Image Type=Normal Scan`. Applied to the `Image Type` field, that rule
conflated **a field that says something disqualifying** with **a field that says nothing
at all**.

Of the 64 excluded sources, 63 record `Copyright: Public Domain` and no image type
whatsoever. They were not judged and found to be re-engravings; they were dropped for a
blank. 61 of the 63 name the person or institution that scanned them
(`{{SibleyScan|1802/17411}}`, `Caprotti`, `Morel`), which is direct evidence that the file
*is* a scan. The reason string `not an original Normal Scan` overstated what had been
established, and reading it at face value is what kept 964 crop rows out of the corpus.

Reinstated for review: **63 sources, 964 rows.** Their evidence is complete — copyright,
publisher, scanner, source and work URLs all present — so the open question is the single
policy one, whether an absent `Image Type` disqualifies a file whose scanner is named. The
owner's decision was that it does not.

Still excluded on its own evidence: **IMSLP257340** (28 rows), whose record states
`Creative Commons Attribution-ShareAlike 4.0` and `Image Type=Typeset`. Neither public
domain nor a scan.

The 126 `needs_manual_review` sources were a separate defect: the resolver searched IMSLP
for the *song* title, while IMSLP files a scan under the *collection* containing it
(`IMSLP44341` is filed under "6 Duets, Op.63", not under "Maiglöckchen und die Blümelein").
`Special:ReverseLookup/<file id>` resolves the file id directly and returned a work page
for every source tried. Note that `Image Type` is not rendered on the work page at all — it
exists only in the wikitext, which IMSLP serves behind a bot check — so that field cannot be
recovered automatically for those 126 and remains a human reading.

Revised position: 25 cleared pending sign-off (565 rows), 63 reinstated (964 rows), 126
under review (2,365 rows), 1 excluded (28 rows).

## Result

| Classification | Source files | Crop rows | Release treatment |
| --- | ---: | ---: | --- |
| `candidate_pd_scan` | 25 | 565 | May proceed to final human/legal provenance sign-off. |
| `exclude` | 64 | 992 | Non-`Normal Scan` source; do not include. |
| `needs_manual_review` | 126 | 2,365 | Record/title/file binding could not be established automatically; do not include. |

The source-disjoint existing v4 split leaves **22 candidate source scores / 499
rows in train** and **3 source scores / 66 rows in validation**. It is adequate as a
small demonstrator/release seed, not a replacement for the 3,922-row internal v4
corpus.

The machine-readable, source-by-source ledger is
[`lieder-rebuild/imslp_rights_audit_v4.csv`](../lieder-rebuild/imslp_rights_audit_v4.csv)
with matching JSONL. It records the work, exact file name, uploader, submission
date, publisher/edition string, IMSLP copyright field, status, reason, and stable
`Special:ImagefromIndex` link. Cached official API responses live outside release
artifacts in `.cache/lieder-rights/imslp-api/` so the result can be reproduced.

## Retained candidate subset

| IMSLP file | Work | Edition evidence from exact IMSLP record |
| --- | --- | --- |
| 10416 | Schubert — *Das Rosenband*, D.280 | Peters, 1895 |
| 10602 | Schubert — *Die Forelle*, D.550 | Peters, 1895 |
| 11784 | Schubert — *Fischerweise*, D.881 | Edition Peters, c.1905–10 |
| 122258 | Poldowski — *Dans une musette* | J. & W. Chester, 1918 |
| 12291 | Schubert — *Ständchen*, D.889 | Peters, 1895 |
| 12778 | Schubert — *Jägers Liebeslied*, D.909 | Peters, 1895 |
| 13971 | Schubert — *Gretchen am Spinnrade*, D.118 | Peters, 1894 |
| 140920 | Chaminade — *L’été* | Joseph Williams, plate N8192 |
| 154060 | Chaminade — *Auprès de ma mie* | G. Schirmer, 1893 |
| 154070 | Chaminade — *L’amour captif* | G. Schirmer, 1895 |
| 154110 | Chaminade — *Aubade* | G. Schirmer, plates 12990–91 |
| 154140 | Chaminade — *Amoroso* | G. Schirmer, 1893 |
| 154142 | Chaminade — *Amour d’automne* | G. Schirmer, 1893 |
| 154144 | Chaminade — *Berceuse* | G. Schirmer, 1894 |
| 16259 | Schubert — *Himmelsfunken*, D.651 | Peters, 1895 |
| 183800 | Barnby — *The Beggar Maid* | Harper & Brothers, 1880 |
| 185782 | Bridge — *The Violets Blue*, H.69 | Winthrop Rogers, 1916 |
| 285344 | Lehmann — *Bouton de Rose* | Boosey & Co., 1899 |
| 33023 | Elgar — *Is She Not Passing Fair?* | Boosey & Co., 1908 |
| 434340 | Munktell — *Österns natt* | Abr. Lundquists, c.1894 |
| 507203 | Ferrari — *Lontan dagli occhi*, Op.17 | Johann André, 1875 |
| 60848 | Schubert — *Die Sterne*, D.939 | Breitkopf, 1895 |
| 74369 | Poldowski — *L’heure exquise* | Charles W. Homeyer, 1917 |
| 83316 | Bizet — *Guitare* | Heugel, 1866 |
| 83319 | Bizet — *Sonnet* | Heugel, 1866 |

The exact source link and raw IMSLP label for every table entry are in the ledger;
do not replace these with work-level links in a dataset card.

## Release requirements

1. Materialize files only from the 25 IDs in
   `lieder-rebuild/imslp_release_candidate_v4_score_ids.txt`; generate new indexes
   from that allow-list and run the normal portable-package verification.
2. Ship `source_provenance.csv` in the dataset repository, copied from the retained
   ledger rows, including the retrieval date and source URL.
3. Keep two rights layers separate in the card: OpenScore Lieder symbolic
   transcriptions are CC0; scan-image provenance is the individual IMSLP record’s
   `Public Domain` assertion. Do **not** label the combined repository CC0.
4. Use a conservative Hugging Face repository licence setting such as `other` with
   this rights statement, unless a reviewer confirms that `public-domain` accurately
   expresses the combined package in every target jurisdiction.
5. Perform a final human check of the 25 linked records immediately before upload,
   record any changed/removed source, and retain the full internal corpus privately.

## Reproduction

```bash
python3 -m training.omr_datasets.audit_lieder_imslp_rights \
  --index lieder-rebuild/imslp_train_index_v4_boundary_safe.txt \
  --index lieder-rebuild/imslp_val_index_v4_boundary_safe.txt \
  --scores-yaml .cache/lieder-rights/scores.yaml \
  --cache .cache/lieder-rights/imslp-api \
  --out-jsonl lieder-rebuild/imslp_rights_audit_v4.jsonl \
  --out-csv lieder-rebuild/imslp_rights_audit_v4.csv
```

`training/omr_datasets/audit_lieder_imslp_rights.py` performs no download of score
files and does not alter the corpus. It records exclusions rather than assuming a
missing or mismatched file page is public domain.
