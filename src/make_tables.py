"""Collect every number quoted in the manuscript into results/manuscript_numbers.json and table CSVs.
The manuscript builder reads only this file, so text and results cannot drift apart."""
import argparse, json
from pathlib import Path
import pandas as pd


def r(x, k=3):
    x = float(x)
    return None if x != x else round(x, k)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', type=Path, required=True); a = ap.parse_args()
    R = a.results; N = {}
    m1 = json.loads((R / 'study1_meta.json').read_text())
    s1 = pd.read_csv(R / 'study1_prequential_summary.csv')
    pc = pd.read_csv(R / 'study1_paired_delay_contrast.csv')
    pr = pd.read_csv(R / 'study1_prequential_predictions.csv')
    d0 = pr[pr.delay_days == 0]
    N['s1'] = dict(
        ts_start=m1['timestamp_range'][0][:10], ts_end=m1['timestamp_range'][1][:10],
        deploy_start=m1['deployment_start'][:10], budget=m1['budget'],
        n_deploy=int(s1.n.iloc[0]), fail_deploy=int(s1.failures.iloc[0]),
        n_burn=int(m1['burn_in_fraction'] * 1567), weeks=int(d0[d0.model == 'logistic'].week.nunique()),
        retro={k: {mm: {q: r(vv) for q, vv in v2.items()} for mm, v2 in v.items()} for k, v in m1['retrospective_random_split'].items()},
        rows=[{k: (r(v) if isinstance(v, float) else v) for k, v in row.items()} for row in s1.to_dict('records')],  # NaN -> null
        paired=[{k: (r(v) if isinstance(v, float) else v) for k, v in row.items()} for row in pc.to_dict('records')])
    m2 = json.loads((R / 'study2_meta.json').read_text())
    s2 = pd.read_csv(R / 'study2_summary.csv'); b2 = pd.read_csv(R / 'study2_instant_label_bias.csv')
    dc = pd.read_csv(R / 'study2_bias_decomposition.csv'); ag = pd.read_csv(R / 'study2_ranking_agreement.csv')
    pdiff = pd.read_csv(R / 'study2_policy_differences.csv'); arl = pd.read_csv(R / 'study2_chart_arl.csv')
    rk = pd.read_csv(R / 'study2_cost_delay_rankings.csv')
    N['s2'] = dict(meta=m2,
                   summary=[{k: (r(v, 2) if isinstance(v, float) else v) for k, v in x.items()} for x in s2.to_dict('records')],
                   bias=[{k: (r(v, 2) if isinstance(v, float) else v) for k, v in x.items()} for x in b2.to_dict('records')],
                   decomposition=[{k: (r(v, 2) if isinstance(v, float) else v) for k, v in x.items()} for x in dc.to_dict('records')],
                   policy_diff=[{k: (r(v, 2) if isinstance(v, float) else v) for k, v in x.items()} for x in pdiff.to_dict('records')],
                   arl=[{k: (r(v, 3) if isinstance(v, float) else v) for k, v in x.items()} for x in arl.to_dict('records')],
                   agreement_rate=r(ag.same_best.mean()), n_cells=int(len(ag)),
                   agreement_by_scenario={k: r(v) for k, v in ag.groupby('scenario').same_best.mean().items()},
                   max_regret=r(ag.regret.max(), 2), mean_regret=r(ag.regret.mean(), 3),
                   regret_outside_high_defect=r(ag[ag.scenario != 'high_defect'].regret.max(), 3),
                   rankings=[{k: (r(v, 2) if isinstance(v, float) else v) for k, v in x.items()} for x in rk.to_dict('records')])
    s3d = R / 'study3'
    if (s3d / 'study3_meta.json').exists():
        m3 = json.loads((s3d / 'study3_meta.json').read_text())
        N['s3'] = dict(meta=m3,
                       rows=[{k: (r(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in pd.read_csv(s3d / 'study3_aoi_summary.csv').to_dict('records')],
                       contrast=[{k: (r(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in pd.read_csv(s3d / 'study3_paired_delay_contrast.csv').to_dict('records')])
    if (s3d / 'study3_methods_summary.csv').exists():
        N['s3m'] = [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in pd.read_csv(s3d / 'study3_methods_summary.csv').to_dict('records')]
    if (R / 'ageing' / 'model_ageing.csv').exists():
        N['age'] = [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in pd.read_csv(R / 'ageing' / 'model_ageing.csv').to_dict('records')]
    if (R / 'decomposition' / 'leakage_decomposition.csv').exists():
        N['leak'] = [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in pd.read_csv(R / 'decomposition' / 'leakage_decomposition.csv').to_dict('records')]
    if (R / 'study2_risk_constrained.csv').exists():
        rc = pd.read_csv(R / 'study2_risk_constrained.csv')
        N['rc'] = [{k: (r(v, 3) if isinstance(v, float) else v) for k, v in x.items()} for x in rc[rc.scenario == 'severe_shift'].to_dict('records')]
    if (R / 'study3' / 'study3_online_calibration.csv').exists():
        N['s3o'] = [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in pd.read_csv(R / 'study3' / 'study3_online_calibration.csv').to_dict('records')]
        N['s3o_meta'] = json.loads((R / 'study3' / 'study3_online_meta.json').read_text())
    if (R / 'study2_exposure_latency.csv').exists():
        N['exp'] = [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in pd.read_csv(R / 'study2_exposure_latency.csv').to_dict('records')]
        N['lmax'] = json.loads((R / 'study2_latency_requirement.json').read_text())
    if (R / 'decomposition' / 'leakage_decomposition_seeds_all.csv').exists():
        N['seeds'] = [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in pd.read_csv(R / 'decomposition' / 'leakage_decomposition_seeds_all.csv').to_dict('records')]
    if (R / 'latency_certification.csv').exists():
        N['cert'] = [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in pd.read_csv(R / 'latency_certification.csv').to_dict('records')]
    if (R / 'study3' / 'study3_selective_baselines.csv').exists():
        N['sel'] = [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in pd.read_csv(R / 'study3' / 'study3_selective_baselines.csv').to_dict('records')]
        N['fh'] = [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in pd.read_csv(R / 'study3' / 'study3_finite_horizon.csv').to_dict('records')]
    sd = R / 'sensitivity'
    if (sd / 'sensitivity_simulation.csv').exists():
        ss = pd.read_csv(sd / 'sensitivity_simulation.csv'); sc = pd.read_csv(sd / 'sensitivity_secom.csv')
        N['sens'] = dict(calibration=json.loads((sd / 'sens_chart_calibration.json').read_text()),
                         sim=[{k: (r(v, 3) if isinstance(v, float) else v) for k, v in x.items()} for x in ss.to_dict('records')],
                         secom=[{k: (r(v, 3) if isinstance(v, float) else v) for k, v in x.items()} for x in sc.to_dict('records')])
    (R / 'manuscript_numbers.json').write_text(json.dumps(N, indent=1, allow_nan=False, default=lambda o: None))
    print('wrote', R / 'manuscript_numbers.json')


if __name__ == '__main__':
    main()
