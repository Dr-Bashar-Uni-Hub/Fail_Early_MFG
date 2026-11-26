# Decision-time information in manufacturing quality evaluation

Reproducibility package for the manuscript *“Decision-time information in manufacturing quality evaluation:
deployment replays on two production datasets and a simulation with SPC comparators”*
(submitted to *Quality Engineering*).

## Quick start

```bash
python -m pip install -r requirements.txt     # Python 3.12
python scripts/get_secom.py                   # downloads UCI SECOM (CC BY 4.0) to data/raw/secom.zip
python run_pipeline.py --sensitivity          # all results, tables and figures (~30 min)
python -m pytest tests                        # unit + consistency tests
```

`python run_pipeline.py --quick` runs a reduced smoke test in under a minute.

## What the pipeline produces

| Step | Script | Main outputs (`results/`) | Manuscript |
|---|---|---|---|
| Study 1: SECOM deployment replay under label delay | `src/secom_prequential.py` | `study1_prequential_summary.csv`, `study1_meta.json`, `study1_retrospective_repeats.csv`, `study1_prequential_predictions.csv` | Section 5.1, Table 3, Figure 2 |
| Study 2: simulation with acceptance sampling, CUSUM and EWMA comparators | `src/delay_sim.py` | `study2_summary.csv`, `study2_instant_label_bias.csv`, `study2_bias_decomposition.csv`, `study2_policy_differences.csv`, `study2_cost_delay_rankings.csv`, `study2_ranking_agreement.csv`, `study2_raw.csv.gz` | Sections 5.4–5.6, Figure 5, Supplementary Tables S1 and S3, Figure S1 |
| Study 3: Siemens AOI deployment replay | `src/aoi_replay.py` | `study3/study3_aoi_summary.csv`, `study3/study3_meta.json`, `study3/study3_weekly.csv`, `study3/study3_paired_delay_contrast.csv`, `study3/study3_retrospective.csv` | Section 5.2, Table 4, Figure 3 |
| Study 3 robustness: models and threshold rules | `src/aoi_methods.py` | `study3/study3_methods_summary.csv`, `study3/study3_methods_retrospective.csv` | Section 5.2, Table 5 |
| Model ageing (both datasets) | `src/model_ageing.py` | `ageing/model_ageing.csv` | Section 5.3, Figure 4 |
| Sources of the gap (size-matched C/M/W designs; batch and causal decisions) | `src/leakage_decomposition.py` | `decomposition/leakage_decomposition.csv` | Section 5.4, Table 7 (three seeds: `leakage_decomposition_seed*.csv`) |
| Risk-constrained policy choice | `src/risk_constrained.py` | `study2_risk_constrained.csv` | Section 5.6 |
| Delay-aware online slip threshold (Proposition 1) | `src/aoi_online.py` | `study3/study3_online_calibration.csv`, `study3/study3_online_weekly.csv` | Sections 3.8 and 5.2, Table 6, Table S6 |
| Latency requirement (Proposition 2) | `src/exposure_latency.py` | `study2_exposure_latency.csv`, `study2_latency_requirement.json` | Sections 4.5 and 5.6 |
| Selective feedback with audits, extra baselines, finite horizon (Proposition 3, Corollary 1) | `src/aoi_selective.py` | `study3/study3_selective_baselines.csv`, `study3/study3_finite_horizon.csv` | Sections 3.9 and 5.2, Table S7 |
| Certified latency requirement (Corollary 2) | `src/latency_certification.py` | `latency_certification.csv`, `latency_certification_truth.json` | Sections 4.6 and 5.6, Table S8 |
| Chart operating characteristics | `src/chart_arl.py` | `study2_chart_arl.csv` | Supplementary Table S2 |
| Figures and paired contrasts | `src/make_figures.py` | `figures/*.png`, `study1_paired_delay_contrast.csv` | Figures 1–6 |
| Sensitivity analyses (Section 5.5) | `src/sensitivity.py` | `sensitivity/sensitivity_simulation.csv`, `sensitivity/sensitivity_secom.csv`, `sensitivity/sens_chart_calibration.json` | Section 5.7, Supplementary Tables S4–S5 |
| Every number quoted in the text | `src/make_tables.py` | `manuscript_numbers.json` | all |

`results/run_record.json` stores the seeds, bootstrap sizes, Python version and SHA-256 hashes of the
input data files used for the run.

## Rebuilding the manuscript

`manuscript/build_manuscript.js` (Node.js, `npm install docx`) builds the Word manuscript and reads every result from `results/manuscript_numbers.json`:

```bash
node manuscript/build_manuscript.js . manuscript/manuscript.docx   # also writes manuscript_Supplement.docx
```

## Mathematical screening

`docs/V6_mathematical_screening_report.md` records the novelty screening of seven candidate mathematical directions, the
closest prior art, the two propositions retained (with proofs) and the directions rejected.
`docs/V6.1_QE61-70_gate_report.md` records the independent audit of the propositions, corrections, new results,
negative results and tasks not done.

## Design notes

* **No label leakage (Study 1).** At each weekly refit only records with `timestamp + delay <= window start`
  are used (`available_indices`, tested in `tests/test_core.py`).
* **Common random numbers (Study 2).** All policies and delays for a given seed use the same item-level
  random stream; differences are paired.
* **Decisions do not depend on costs.** Net value for any cost pair is computed exactly from recorded counts.
* **Report timing.** A batch inspected in period *t* is first usable at period *t + 1 + L*.

## Data

SECOM: McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C54305 (CC BY 4.0). The raw files are not committed; `scripts/get_secom.py`
downloads them.

## Study 3: Siemens AOI false-call data

`src/aoi_replay.py` applies the deployment replay to the Siemens automated-optical-inspection dataset
(Pfab et al. 2024, *Data in Brief*, doi:10.1016/j.dib.2024.110110). Download `dataset.csv` from Mendeley Data
(doi:10.17632/99jzmh9658.1); it is not redistributed here. Then run

```bash
python run_pipeline.py --aoi-csv path/to/dataset.csv --sensitivity
```

Outputs go to `results/study3/` (Section 5.2, Tables 4–5, Figure 3) and `results/ageing/` (Section 5.3, Figure 4). The decision is to release AOI alarms without
manual inspection; outcomes are slip (share of true defects released) and volume reduction (share of false calls
released) for a threshold set to 1% slip on recent labelled history, and slip when the 40% lowest-risk alarms
are released.

## Earlier versions

`archive/legacy_v3/` holds the code and outputs of the earlier (v3) analysis for transparency. It is not
used by the manuscript. See `CHANGELOG.md`.

## License and citation

Code: MIT (see `LICENSE`). Please cite the article and the archived release (see `CITATION.cff`).
