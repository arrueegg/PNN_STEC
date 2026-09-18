# Where every number in the revised manuscript comes from

**Written 2026-09-15 by session `pnn-stec-eb` as a handover.** `STEC_Modelling/` is
gitignored, so the manuscript has no version history, and up to six Claude sessions can
reach this checkout. Cross-session messages are held for approval and session addresses
change, so this file — not a message — is the durable record.

**Ownership as of writing:** `pnn-stec-c4` owns `STEC_Modelling/` (the `.tex`, the figures
and the build) and `docs/revision/response_to_reviewers.md` / `evidence_summary.md`.
`pnn-stec-eb` owns the `stec/` pipeline and stops at the manuscript boundary.

**State of the manuscript:** `PNN_main_revised.tex` last written 17:37 by `pnn-stec-eb`;
compiled 18:11 by `pnn-stec-c4` with latexmk. 31 pages, 0 errors, 0 undefined references,
0 undefined citations. Pre-edit md5 `282cfad502354e388d609affd01a8822`, current md5
`be1b14f31015a5e9fc241ee86a120638`.

**Read the value from the path, never from this table.** Every number below is a snapshot
taken 2026-09-15 ~18:35. All paths are relative to `multiday_results/analyses/`. All 48
pipeline stages were up to date when these were read.

```
MANUSCRIPT CLAIM                       VALUE (read 2026-09-15 18:3x)                                  SOURCE
Table 3 Direct STEC Model              6.87 [6.09-7.66] | MAE 3.87 | R2 0.97 | P95 13.12              daily_metrics/rebuilt/summary.csv
Table 3 IGS GIM                        8.32 [7.63-8.97] | MAE 5.28 | R2 0.96 | P95 15.71              daily_metrics/rebuilt/summary.csv
Table 3 Pretrained STEC                12.19 [10.27-15.49] | MAE 8.17 | R2 0.90 | P95 24.14           daily_metrics/rebuilt/summary.csv
Table 3 VTEC + Mapping                 8.97 [7.79-9.85] | MAE 5.12 | R2 0.95 | P95 16.58              daily_metrics/rebuilt/summary.csv
Table 4 Direct STEC Model              14.00 [11.85-16.91] | MAE 8.45 | R2 0.86 | P95 31.36           daily_metrics/rebuilt/summary.csv
Table 4 IGS GIM                        15.07 [13.09-17.19] | MAE 10.09 | R2 0.84 | P95 33.39          daily_metrics/rebuilt/summary.csv
Table 4 Pretrained STEC                16.47 [14.24-19.54] | MAE 10.92 | R2 0.81 | P95 35.05          daily_metrics/rebuilt/summary.csv
Table 4 VTEC + Mapping                 13.41 [11.23-15.54] | MAE 8.09 | R2 0.88 | P95 28.85           daily_metrics/rebuilt/summary.csv
Table 5 Direct STEC (own)              2.52 [1.50-4.48]                                               dstec_evaluation/rebuilt/summary.csv
Table 5 IGS GIM (own)                  4.10 [2.53-6.71]                                               dstec_evaluation/rebuilt/summary.csv
Table 5 VTEC+Map (own)                 4.67 [2.80-7.75]                                               dstec_evaluation/rebuilt/summary.csv
Table 5 Pretrained (own)               6.31 [3.58-10.71]                                              dstec_evaluation/rebuilt/summary.csv
Table 5 Direct STEC (Madrigal)         3.90 [2.18-7.46]                                               dstec_evaluation_madrigal/rebuilt/summary.csv
Table 5 IGS GIM (Madrigal)             5.32 [3.22-9.17]                                               dstec_evaluation_madrigal/rebuilt/summary.csv
Table 5 VTEC+Map (Madrigal)            6.00 [3.58-10.44]                                              dstec_evaluation_madrigal/rebuilt/summary.csv
Table 5 Pretrained (Madrigal)          7.38 [4.10-12.95]                                              dstec_evaluation_madrigal/rebuilt/summary.csv
Sec 4.4 activity: normal               STEC 0.834 vs GIM 1.034 = 19.4%; >10m 24 vs 9                  positioning_activity/rebuilt/activity_stratification.csv
Sec 4.4 activity: elevated             STEC 1.125 vs GIM 1.204 = 6.6%; >10m 46 vs 7                   positioning_activity/rebuilt/activity_stratification.csv
Sec 4.4 constellation: all station-days n=10715 sats 15.83/16.54 impr 18.5% >10m 70/15                 constellation_coverage/rebuilt/population_summary.csv
Sec 4.4 constellation: both constellations corrected n=8190 sats 16.38/16.41 impr 22.9% >10m 23/15                  constellation_coverage/rebuilt/population_summary.csv
Sec 4.4 constellation: single constellation corrected n=2525 sats 8.57/17.05 impr -8.2% >10m 47/0                    constellation_coverage/rebuilt/population_summary.csv
Sec 4.2 calibrating factor             median 1.644, range 1.544-1.681, 9 bands                       uncertainty_error_relation/rebuilt/calibrating_factor.csv
```

## Numbers not in the table above

- Madrigal per-station offsets quoted in the Table 4 discussion (+8.76 / +6.51 / +5.69 /
  +4.34 TECU and the four Spearman correlations against absolute geomagnetic latitude):
  `madrigal_method_offset_comparison/rebuilt/per_station_offsets.csv`. That stage went
  stale downstream of the `daily_metrics` column change and was re-run; the offsets came
  back byte-identical, so no manuscript number moved.
- Within-station constellation penalty x1.41, 21 of 23 stations significant:
  `constellation_coverage/rebuilt/within_station_penalty.csv`. It ranges x1.40 to x1.55
  with the `min_days_each` threshold, which is why the manuscript quotes a range and names
  the threshold rather than a bare point estimate.
- CRPS, PIT KS and coverage: `uncertainty_calibration/rebuilt/pretrained_stec_own/` and
  `.../finetuned_stec_own/`. **These are two different models.** The manuscript keeps them
  in separate subsections on purpose; do not merge them.
- The 7.6 cm satellite-code-bias figure in the positioning methods was measured by an A/B
  PPPx run on one station-day (WARK, DOY 122) and is **not** backed by a declared stage.
  It is the one number in the manuscript with no artifact behind it.

## Open items at handover

1. **Embedded figures may be stale.** `STEC_Modelling/Figure{4,5,8}.png` date from
   2026-08-18 and `Figure{10,11}.png` from 09-14, while their generating stages have since
   re-run. If the copy from `plots/` into `STEC_Modelling/` is manual, the compiled PDF may
   show old plots while every number verifies — the most plausible route by which a wrong
   figure reaches submission.
2. **Tables renumbered.** The new combined dSTEC table is Table 5, so the positioning
   summary is now 6, components 7, weighting ablation 8. `trackchanges`/`soul` cannot take
   `\ref` (nor a `tabular`), so table references in this document are literal numbers by
   necessity; inserting another table means updating them by hand again.
3. **Supporting documents unaudited.** `response_to_reviewers.md` and
   `evidence_summary.md` have not been checked against today's numbers. The
   recovered/original framing is gone from the manuscript, Tables 3/4 now report medians,
   and the tables renumbered.
4. **No transcription check.** Numbers were copied into the `.tex` by hand. The pattern for
   fixing this already exists: `paper_tables_manuscript_rows_are_backed` in
   `stec/pipeline/stages.py` reads the real `.tex` and checks Table 2's *rows* against the
   CSV, and its docstring names value-checking as the gap it deliberately does not cover.
   Extending it to Tables 3/4/5 values would make this file unnecessary.
5. **Elevation cutoff.** The STEC-domain results use a 5 degree floor and PPPx a 7 degree
   mask, so the two experiments are not evaluated over the same observation population.
   Documented in `stec/analysis/divergences.py`; still unstated in the manuscript.
