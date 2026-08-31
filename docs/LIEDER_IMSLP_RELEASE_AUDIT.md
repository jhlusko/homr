# Lieder IMSLP scan release audit

**Status: a small, conservative public-release candidate exists; the full scanned
Lieder corpus is not cleared for redistribution.** This audit checked every one of
the 215 IMSLP file IDs used by the v4 boundary-safe corpus on 2026-08-31.

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
