"""Sensitivity analyses requested by likely reviewer questions.

Study 2 (simulation) variants, each re-running the full delay x policy x scenario design:
  base            : main-text settings
  spec_tuned      : CUSUM p1 = 0.12 and EWMA lambda = 0.1, limits calibrated to in-control ARL ~ 400 batches
  long_horizon    : H = 40 periods, shift after period 16
  small_batch     : B = 10 items per period
  strict_plan     : acceptance plan n = 40, c = 2
  near_spec_shift : severe shift to p = 0.15 instead of 0.24
Study 1 (SECOM) variants: burn-in 30% / 50%, inspection budget 5% / 20%, 14-day refit window.
"""
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd
import delay_sim as S
import secom_prequential as P

BASE = dict(H=S.H, B=S.B, SHIFT_T=S.SHIFT_T, P1=S.P1, H_CUSUM=S.H_CUSUM, LAM=S.LAM,
            EWMA_UCL=S.EWMA_UCL, AQL_C=S.AQL_C, SCEN=dict(S.SCEN), JITTER=0, RAMP=0, RHO=0.0, WS_FIX=False)


def set_sim(**kw):
    for k, v in {**BASE, **kw}.items():
        setattr(S, k, v)


def arl0_cusum(h, p1, reps, rng, cap=4000):
    out = []
    for _ in range(reps):
        s = 0.
        for t in range(1, cap + 1):
            k = rng.binomial(S.B, S.P0)
            s = max(0., s + k * math.log(p1 / S.P0) + (S.B - k) * math.log((1 - p1) / (1 - S.P0)))
            if s > h:
                break
        out.append(t)
    return float(np.mean(out))


def arl0_ewma(ucl, lam, reps, rng, cap=4000):
    out = []
    for _ in range(reps):
        z = S.P0
        for t in range(1, cap + 1):
            z = lam * rng.binomial(S.B, S.P0) / S.B + (1 - lam) * z
            if z > ucl:
                break
        out.append(t)
    return float(np.mean(out))


def calibrate(target=400., reps=3000):
    """Bisection on the chart limit so that the in-control ARL is close to the target."""
    rng = np.random.default_rng(11)
    lo, hi = 1.0, 8.0
    for _ in range(14):
        mid = (lo + hi) / 2
        (lo, hi) = (mid, hi) if arl0_cusum(mid, .12, reps, rng) < target else (lo, mid)
    h = (lo + hi) / 2
    lo, hi = .04, .12
    for _ in range(14):
        mid = (lo + hi) / 2
        (lo, hi) = (mid, hi) if arl0_ewma(mid, .1, reps, rng) < target else (lo, mid)
    ucl = (lo + hi) / 2
    rng = np.random.default_rng(12)
    return dict(cusum_h=h, cusum_arl0=arl0_cusum(h, .12, 10000, rng), ewma_ucl=ucl, ewma_arl0=arl0_ewma(ucl, .1, 10000, rng))


def sim_variants(cal):
    near = dict(BASE['SCEN']); near['severe_shift'] = (.035, .15)
    return {
        'base': {},
        'spec_tuned': dict(P1=.12, H_CUSUM=cal['cusum_h'], LAM=.1, EWMA_UCL=cal['ewma_ucl']),
        'long_horizon': dict(H=40, SHIFT_T=16),
        'small_batch': dict(B=10),
        'strict_plan': dict(AQL_C=2),
        'near_spec_shift': dict(SCEN=near),
        'variable_delay': dict(JITTER=1),
        'gradual_drift': dict(RAMP=6),
        'correlated_defects': dict(RHO=.05),
        'ws_corrected': dict(WS_FIX=True),
    }


def run_sim(out, seeds, only=None):
    cf = out / 'sens_chart_calibration.json'
    if cf.exists():
        cal = json.loads(cf.read_text())
    else:
        cal = calibrate(); cf.write_text(json.dumps(cal, indent=2))
    rows = []
    for name, kw in sim_variants(cal).items():
        if only and name not in only:
            continue
        set_sim(**kw)
        d = out / f'sens_sim_{name}'; d.mkdir(parents=True, exist_ok=True)
        df = S.run(seeds); S.analyse(df, d)
        b = pd.read_csv(d / 'study2_instant_label_bias.csv'); ag = pd.read_csv(d / 'study2_ranking_agreement.csv')
        sm = pd.read_csv(d / 'study2_summary.csv')
        g = lambda sc, L, p, c: float(sm[(sm.scenario == sc) & (sm.delay == L) & (sm.policy == p)][c].iloc[0])
        b3 = b[(b.delay == 3) & b.scenario.isin(['nominal', 'severe_shift'])]
        rows.append(dict(variant=name, seeds=seeds,
                         bias3_min=b3.bias.min(), bias3_max=b3.bias.max(),
                         bias3_severe_AC=float(b3[(b3.scenario == 'severe_shift') & (b3.policy == 'AC')].bias.iloc[0]),
                         agreement=ag.same_best.mean(), max_regret=ag.regret.max(),
                         max_regret_excl_high_defect=ag[ag.scenario != 'high_defect'].regret.max(),
                         moderate_rollback_AC=g('moderate_shift', 0, 'AC', 'rollback_rate'),
                         nominal_rollback_AC=g('nominal', 0, 'AC', 'rollback_rate'),
                         severe_oos_FX=g('severe_shift', 0, 'FX', 'mean_oos_periods'),
                         severe_oos_AC=g('severe_shift', 0, 'AC', 'mean_oos_periods'),
                         severe_net_FX=g('severe_shift', 0, 'FX', 'mean_net'),
                         severe_net_AC=g('severe_shift', 0, 'AC', 'mean_net'),
                         highdef_net_WS_L0=g('high_defect', 0, 'WS', 'mean_net'),
                         highdef_net_WS_L3=g('high_defect', 3, 'WS', 'mean_net')))
    set_sim()
    new = pd.DataFrame(rows); f = out / 'sensitivity_simulation.csv'
    if only and f.exists():
        old = pd.read_csv(f); new = pd.concat([old[~old.variant.isin(new.variant)], new], ignore_index=True)
    new.to_csv(f, index=False)
    print(new.round(3).to_string())


def run_secom(zip_path, out, boot, repeats):
    X, y, ts = P.load(zip_path)
    variants = {'base': {}, 'burn_in_30': dict(BURN_IN=.30), 'burn_in_50': dict(BURN_IN=.50),
                'budget_05': dict(BUDGET=.05), 'budget_20': dict(BUDGET=.20), 'window_14d': dict(WINDOW=14)}
    rng = np.random.default_rng(20261123); rows = []
    for name, kw in variants.items():
        P.BURN_IN, P.BUDGET = kw.get('BURN_IN', .40), kw.get('BUDGET', .10)
        window = kw.get('WINDOW', 7)
        retro, _ = P.retrospective(X, y, repeats=repeats)
        for L in (0, 14):
            df = P.prequential(X, y, ts, L, window_days=window)
            for r in P.summarise(df, rng, boot):
                rows.append(dict(variant=name, burn_in=P.BURN_IN, budget=P.BUDGET, window_days=window, delay_days=L,
                                 model=r['model'], n_replay=r['n'], failures=r['failures'],
                                 replay_recall=r['recall_at_budget'], replay_lo=r['recall_lo'], replay_hi=r['recall_hi'],
                                 replay_auc=r['pooled_roc_auc'],
                                 retro_recall=retro[r['model']]['recall_at_budget']['mean'],
                                 retro_auc=retro[r['model']]['roc_auc']['mean'], random_recall=P.BUDGET))
        print(name, 'done', flush=True)
    P.BURN_IN, P.BUDGET = .40, .10
    pd.DataFrame(rows).to_csv(out / 'sensitivity_secom.csv', index=False)
    print(pd.DataFrame(rows).round(3).to_string())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--part', choices=['sim', 'secom', 'all'], default='all')
    ap.add_argument('--secom-zip', type=Path); ap.add_argument('--seeds', type=int, default=500)
    ap.add_argument('--boot', type=int, default=500); ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--only', nargs='*', help='run only these simulation variants (merged into the existing table)')
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    if a.part in ('sim', 'all'):
        run_sim(a.output, a.seeds, a.only)
    if a.part in ('secom', 'all'):
        run_secom(a.secom_zip, a.output, a.boot, a.repeats)


if __name__ == '__main__':
    main()
