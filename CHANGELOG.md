# Changelog

## 6.2.0 (2026-11-25) — submission version (no new research)
* Proposition 3 clarified: pending audit outcomes at the horizon; assumption relaxed to predictable scores.
* Qualifications: Clopper-Pearson stepwise rule valid only under exchangeability; certification guarantee vs empirical
  coverage (EWMA); C/M/W calibration differences; SECOM delay effects stated with their uncertainty (Results, Discussion,
  Conclusions); abstract audit claim (slip above the 1% target). Outdated Section 6.4 seed statement corrected.
* Initial label availability stated explicitly.
* Figure 1 redrawn; all figures also exported as vector PDF.
* Audited-release and certification methods moved to Supplementary Appendix S2.
* Draft AI-use statement (highlighted for author confirmation).
* Verification: 15 unit tests pass; quick end-to-end run (`run_pipeline.py --quick`, SECOM + simulation) completed
  without errors on 2026-11-25; AOI steps not re-run for this version (code unchanged). See docs/V6.2_QE71-75_final_report.md.

## 6.1.0 (2026-11-25)
* Fixed: SECOM initial-history count reported as 627 (rounding); the analysis used 626 (626 + 941 = 1,567). Unit test added.
* Proposition 1 audited (variable/out-of-order/burst delays; tightened constant); Proposition 2 assumptions corrected
  (same monitoring state at the shift), empty/unbounded feasible sets handled.
* `src/aoi_selective.py`: randomized post-release audits with inverse-propensity updates (Proposition 3, proof in
  Appendix S1), q = 0 impossibility, sliding-window / exponentially weighted / Clopper-Pearson window baselines on identical
  scores, finite-horizon pending-confirmation bound (Corollary 1; negative result reported).
* `src/latency_certification.py`: DKW-based simultaneous certification of the latency requirement vs plug-in,
  Hoeffding and empirical-Bernstein (Corollary 2; Table S8).
* Abstract wording on SECOM delay effects corrected. Gate report in `docs/`.

## 6.0.0 (2026-11-25)
* Mathematical screening of seven proposed directions (`docs/V6_mathematical_screening_report.md`); no strong original
  mathematical contribution survived the prior-art check, two applied propositions did. Title and positioning unchanged.
* `src/aoi_online.py`: delay-aware online release threshold (quantile tracking with action-dependent feedback delay);
  Proposition 1 with proof (Appendix S1), adversarial unit test; Section 3.8, Table 6, Table S6.
* `src/exposure_latency.py`: latency requirement for an exposure limit; Proposition 2 with proof; validated for all
  policies; Sections 4.5 and 5.6.
* C/M/W decomposition repeated with two further independent seeds.
* Default behaviour of all V5.0 analyses unchanged.

## 5.0.0 (2026-11-25)
* Retitled: “Decision-time information in manufacturing quality evaluation …”.
* `src/leakage_decomposition.py`: size-matched designs C (earlier units only), M (other weeks, earlier or later) and W (may include
  same-week units) on the same evaluated half-weeks; batch and strictly causal decisions; lift; paired day-block bootstrap (Section 5.4, Table 6).
* `src/risk_constrained.py`: value-maximizing policy subject to an out-of-specification limit (Section 5.6).
* Simulator extensions (defaults unchanged, verified identical): variable delay, gradual drift, beta-binomial correlated defects,
  corrected Wilson-score rule (wait for pending reports after five pilots); four new sensitivity variants (Table S4).
* Wording: delays are scenarios; ageing is an empirically supported mechanism, not a law; the confidence-bound rule reduces risk but
  does not guarantee compliance; random-split figures are optimistic rather than upper bounds. Box 1 gains a decision-timing item.

## 4.3.0 (2026-11-23)
* `src/aoi_methods.py`: Study 3 with random forest and logistic regression, and with three alternative threshold rules
  (last-7-day calibration, 0.5% target, one-sided 95% Clopper–Pearson upper-bound rule) (Section 5.2, Table 5).
* `src/model_ageing.py`: model-ageing curves for SECOM and AOI explaining when label delay matters (Section 5.3, Figure 4).
* Manuscript split into main text and supplementary material (Tables S1–S5, Figure S1); Box 1 gains a model-ageing item.

## 4.2.0 (2026-11-23)
* Study 3 run on the real Siemens AOI data (440,274 alarms): threshold rule for 1% slip and fixed 40% release,
  weekly replay with delays 0–14 days, frozen model, 10 random splits, paired day-block bootstrap (Section 5.2).
* `run_pipeline.py --aoi-csv` option; Study 3 figure; unit test for the slip threshold.
* Manuscript retitled; tables and figures renumbered.

## 4.1.0 (2026-11-23)
* Results regenerated from the official UCI SECOM archive (identical values to 4.0.0; input SHA-256 in `results/run_record.json`).
* New `src/sensitivity.py` (Section 5.5): chart tuning to the 0.12 specification with calibrated in-control ARL, 40-period horizon,
  10-item batches, stricter acceptance plan (c = 2), near-specification shift (p = 0.15); SECOM burn-in 30%/50%, budget 5%/20%, 14-day refit window.
* Fixed: acceptance-plan pilot rule now samples n = 40 items for any batch size (previously two pilots, which is identical only for 20-item batches; main results unchanged, verified).
* New `src/aoi_replay.py` for the Siemens AOI dataset (optional Study 3).

## 4.0.0 (2026-11-23)
* New Study 1: SECOM deployment replay with label delays of 0, 1, 3, 7 and 14 days, weekly refits on
  released labels only, top-10% inspection budget, day-block bootstrap; retrospective benchmark from
  50 repeated stratified random splits.
* New Study 2: 20-period simulator with matched actions and information; four policies including a
  single acceptance-sampling plan with binomial CUSUM or EWMA monitoring; delays 0–4; 1,000 paired seeds;
  exact cost-grid recomputation; bias decomposition; ranking regret; chart ARLs.
* Fixed: report-release timing (earlier code treated delays 0 and 1 identically in the new simulator draft).
* Single entry point `run_pipeline.py`; manuscript numbers generated into `results/manuscript_numbers.json`.
* Pinned dependencies; unit tests; removed hard-coded paths.
* Removed from the public tree: internal working drafts, a script that edited the manuscript file,
  a duplicated framework module.
* Earlier analyses moved to `archive/legacy_v3/` (not used by the manuscript).

## 3.x — earlier working versions (archived)
