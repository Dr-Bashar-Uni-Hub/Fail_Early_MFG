"""Study 2: controlled simulation of pilot -> deploy -> monitor -> rollback decisions under report delay.

All policies share: the same per-item random stream (common random numbers), the same action set
(pilot, deploy, abandon, continue, rollback), the same released-report information, and the same
cost accounting. They differ only in decision rules:

  WS   Wilson-score heuristic (the earlier 'queue guard'): deploy when >=40 released items and the 95%
       Wilson upper bound <= 0.18; roll back when the cumulative released defect rate > 0.12.
  FX   Fixed two-pilot rule: deploy if the released rate <= 0.12 after two pilot batches; roll back when
       the cumulative released rate > 0.20.
  AC   Single acceptance-sampling plan for deployment (n = 40, c = 3) and a binomial CUSUM on released
       post-deployment batches (p0 = 0.035, p1 = 0.20, h = 4.0) for rollback.
  AE   Same acceptance plan; EWMA chart on released batch fraction defective (lambda = 0.2, 3-sigma limit
       at p0 = 0.035) for rollback.

Decisions never depend on cost parameters, so net value for any (defect penalty, rollback cost)
pair is computed exactly from recorded counts. A 'delay' of L periods means a batch inspected in
period t is released at the start of period t+1+L (L = 0: available at the next decision). Evaluation bias of an instantaneous-label analysis
is V(L=0) - V(L) on identical seeds.
"""
import argparse, math, json
from pathlib import Path
import numpy as np, pandas as pd

H, B, SHIFT_T = 20, 20, 8
SCEN = {  # name: (defect prob before shift, after shift)
    'nominal': (.035, .035),
    'moderate_shift': (.035, .10),
    'severe_shift': (.035, .24),
    'high_defect': (.22, .22),
}
POL = ('WS', 'FX', 'AC', 'AE')
DELAYS = (0, 1, 2, 3, 4)
SPEC = .12                 # quality limit used only for retrospective out-of-spec accounting
REVENUE, PILOT_COST, DEPLOY_COST = 8.0, .5, 3.0
PENALTIES, ROLLBACKS = (15, 35, 70), (2, 6, 18)
BASE_PEN, BASE_RB = 35, 6
P0, P1, H_CUSUM = .035, .20, 4.0
LAM = .2
EWMA_UCL = P0 + 3 * math.sqrt(P0 * (1 - P0) / B) * math.sqrt(LAM / (2 - LAM))
AQL_N, AQL_C = 40, 3
# Extensions (defaults reproduce the main analysis exactly)
JITTER = 0      # variable delay: each batch's delay is max(0, L + U{-JITTER..JITTER})
RAMP = 0        # gradual drift: p moves linearly to its post-shift value over RAMP periods
RHO = 0.0       # correlated defects: batch defect probability ~ Beta with mean p and intra-batch correlation RHO
WS_FIX = False  # corrected Wilson-score rule: after 5 pilots stop piloting, wait for pending reports, then decide


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return 1.
    r = k / n
    return (r + z * z / (2 * n) + z * math.sqrt(r * (1 - r) / n + z * z / (4 * n * n))) / (1 + z * z / n)


class Policy:
    def __init__(self, name):
        self.name = name; self.n = self.k = 0; self.pn = self.pk = 0; self.pilots = 0
        self.s = 0.; self.z = P0

    def observe(self, k, deployed_batch):
        self.n += B; self.k += k
        if not deployed_batch:
            self.pn += B; self.pk += k
        else:
            inc = k * math.log(P1 / P0) + (B - k) * math.log((1 - P1) / (1 - P0))
            self.s = max(0., self.s + inc)
            self.z = LAM * k / B + (1 - LAM) * self.z

    def pilot_decision(self, pending):
        n, k = self.n, self.k
        if self.name == 'WS':
            if n >= 40 and wilson_upper(k, n) <= .18:
                return 'deploy'
            if WS_FIX and self.pilots >= 5:
                return 'wait' if pending else 'abandon'     # stop piloting; decide once all reports are in
            return 'abandon' if self.pilots >= 5 and not pending else 'pilot'
        if self.name == 'FX':
            if self.pilots >= 2 and n >= 40:
                return 'deploy' if k / n <= .12 else 'abandon'
            return 'pilot'
        # acceptance plan (AC, AE): decide as soon as the 40-item sample is released
        if self.pn >= AQL_N:
            return 'deploy' if self.pk <= AQL_C else 'abandon'
        return 'pilot' if self.pilots * B < AQL_N else 'wait'   # sample AQL_N items, then wait for results

    def monitor_decision(self):
        if self.name == 'WS':
            return 'rollback' if self.n >= 40 and self.k / self.n > .12 else 'continue'
        if self.name == 'FX':
            return 'rollback' if self.n >= 40 and self.k / self.n > .20 else 'continue'
        if self.name == 'AC':
            return 'rollback' if self.s > H_CUSUM else 'continue'
        return 'rollback' if self.z > EWMA_UCL else 'continue'


def simulate(u, p_path, pol, delay, jit=None, z=None):
    """u: (H, B) uniforms shared by all policies for a seed; p_path: defect probability per period."""
    P = Policy(pol); mode = 'pilot'; pending = []
    rec = dict(pilots=0, deployed=0, rolled=0, prod_periods=0, defects=0, oos_periods=0,
               deploy_t=-1, stop_t=-1, waits=0)
    for t in range(H):
        for item in [x for x in pending if x[0] <= t]:
            P.observe(item[1], item[2])
        pending = [x for x in pending if x[0] > t]
        if mode == 'pilot':
            a = P.pilot_decision(bool(pending))
        else:
            a = P.monitor_decision()
        if a == 'abandon':
            rec['stop_t'] = t; break
        if a == 'rollback':
            rec['rolled'] = 1; rec['stop_t'] = t; break
        if a == 'wait':
            rec['waits'] += 1; continue
        if a == 'deploy':
            mode = 'deployed'; rec['deployed'] = 1; rec['deploy_t'] = t
        p_t = p_path[t]
        if RHO > 0 and z is not None and 0 < p_t < 1:
            from scipy.stats import beta as _beta
            p_t = float(_beta.ppf(z[t], p_t * (1 - RHO) / RHO, (1 - p_t) * (1 - RHO) / RHO))
        k = int((u[t] < p_t).sum())
        if a == 'pilot':
            P.pilots += 1; rec['pilots'] += 1
        else:
            rec['prod_periods'] += 1; rec['defects'] += k; rec['oos_periods'] += int(p_path[t] > SPEC)
        d_t = delay if jit is None else max(0, delay + int(jit[t]))
        pending.append((t + 1 + d_t, k, mode == 'deployed' and a != 'pilot'))
    return rec


def net_value(df, pen, rb):
    return (REVENUE * df.prod_periods - pen * df.defects / B - PILOT_COST * df.pilots
            - DEPLOY_COST * df.deployed - rb * df.rolled)


def run(seeds):
    rows = []
    for sc, (pa, pb) in SCEN.items():
        p_path = np.array([pa if t < SHIFT_T else (pb if RAMP <= 0 else pa + (pb - pa) * min(1., (t - SHIFT_T + 1) / RAMP)) for t in range(H)])
        for seed in range(seeds):
            u = np.random.default_rng([20261123, seed]).random((H, B))
            r2 = np.random.default_rng([20261124, seed])
            jit = r2.integers(-JITTER, JITTER + 1, H) if JITTER else None
            z = r2.random(H) if RHO > 0 else None
            for L in DELAYS:
                for pol in POL:
                    rows.append(dict(scenario=sc, seed=seed, delay=L, policy=pol, **simulate(u, p_path, pol, L, jit, z)))
    df = pd.DataFrame(rows)
    df['net_base'] = net_value(df, BASE_PEN, BASE_RB)
    return df


def boot_mean_ci(d, rng, B=2000):
    d = np.asarray(d, float); n = len(d)
    m = rng.integers(0, n, (B, n))
    bs = d[m].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analyse(df, out):
    rng = np.random.default_rng(7)
    # 1) Summary by scenario, delay, policy (base costs)
    summ = df.groupby(['scenario', 'delay', 'policy']).agg(
        n=('seed', 'size'), mean_net=('net_base', 'mean'), deploy_rate=('deployed', 'mean'),
        rollback_rate=('rolled', 'mean'), mean_prod_periods=('prod_periods', 'mean'),
        mean_defects=('defects', 'mean'), mean_oos_periods=('oos_periods', 'mean')).reset_index()
    summ.to_csv(out / 'study2_summary.csv', index=False)
    # 2) Instantaneous-label evaluation bias: V(L=0) - V(L), paired on seed
    w = df.pivot_table(index=['scenario', 'policy', 'seed'], columns='delay', values='net_base').reset_index()
    bias = []
    for (sc, pol), g in w.groupby(['scenario', 'policy']):
        for L in DELAYS[1:]:
            m, lo, hi = boot_mean_ci(g[0] - g[L], rng)
            base = g[L].mean()
            bias.append(dict(scenario=sc, policy=pol, delay=L, value_true_delay=base, value_instant=g[0].mean(),
                             bias=m, bias_lo=lo, bias_hi=hi,
                             rel_bias_pct=100 * m / abs(base) if abs(base) > 1e-9 else np.nan))
    pd.DataFrame(bias).to_csv(out / 'study2_instant_label_bias.csv', index=False)
    # 3) Paired policy differences (vs WS) at each delay, base costs
    pdiff = []
    w2 = df.pivot_table(index=['scenario', 'delay', 'seed'], columns='policy', values='net_base').reset_index()
    for (sc, L), g in w2.groupby(['scenario', 'delay']):
        for pol in ('FX', 'AC', 'AE'):
            m, lo, hi = boot_mean_ci(g[pol] - g['WS'], rng)
            pdiff.append(dict(scenario=sc, delay=L, comparison=f'{pol} minus WS', mean_diff=m, lo=lo, hi=hi))
    pd.DataFrame(pdiff).to_csv(out / 'study2_policy_differences.csv', index=False)
    # 4) Ranking under every cost pair and delay; does the instant-label ranking agree with the true-delay ranking?
    grid = []
    for pen in PENALTIES:
        for rb in ROLLBACKS:
            v = df.assign(net=net_value(df, pen, rb)).groupby(['scenario', 'delay', 'policy']).net.mean().reset_index()
            for (sc, L), g in v.groupby(['scenario', 'delay']):
                g = g.sort_values('net', ascending=False)
                grid.append(dict(scenario=sc, delay=L, penalty=pen, rollback_cost=rb,
                                 best=g.policy.iloc[0], ranking='>'.join(g.policy),
                                 **{f'net_{p}': float(g.set_index('policy').net[p]) for p in POL}))
    grid = pd.DataFrame(grid); grid.to_csv(out / 'study2_cost_delay_rankings.csv', index=False)
    agree = []
    for (sc, pen, rb), g in grid.groupby(['scenario', 'penalty', 'rollback_cost']):
        b0 = g[g.delay == 0].best.iloc[0]
        for L in DELAYS[1:]:
            agree.append(dict(scenario=sc, penalty=pen, rollback_cost=rb, delay=L,
                              best_instant=b0, best_true=g[g.delay == L].best.iloc[0],
                              same_best=b0 == g[g.delay == L].best.iloc[0]))
    agree = pd.DataFrame(agree)
    # regret: true-delay value lost by choosing the instant-label winner
    reg = []
    for _, r in agree.iterrows():
        g = grid[(grid.scenario == r.scenario) & (grid.penalty == r.penalty) & (grid.rollback_cost == r.rollback_cost) & (grid.delay == r.delay)].iloc[0]
        reg.append(g[f'net_{r.best_true}'] - g[f'net_{r.best_instant}'])
    agree['regret'] = reg
    agree.to_csv(out / 'study2_ranking_agreement.csv', index=False)
    # 6) decomposition of instant-label bias into components (base costs), V(0)-V(L)
    comp = df.assign(revenue=REVENUE * df.prod_periods, defect_cost=BASE_PEN * df.defects / B,
                     pilot_cost=PILOT_COST * df.pilots, deploy_cost=DEPLOY_COST * df.deployed,
                     rollback_cost=BASE_RB * df.rolled)
    cm = comp.groupby(['scenario', 'policy', 'delay'])[['revenue', 'defect_cost', 'pilot_cost', 'deploy_cost', 'rollback_cost', 'net_base', 'oos_periods', 'defects']].mean().reset_index()
    dec = []
    for (sc, pol), g in cm.groupby(['scenario', 'policy']):
        g = g.set_index('delay')
        for L in DELAYS[1:]:
            dec.append(dict(scenario=sc, policy=pol, delay=L,
                            lost_revenue=g.revenue[0] - g.revenue[L],
                            extra_defect_cost=g.defect_cost[L] - g.defect_cost[0],
                            extra_pilot_cost=g.pilot_cost[L] - g.pilot_cost[0],
                            other=(g.deploy_cost[L] - g.deploy_cost[0]) + (g.rollback_cost[L] - g.rollback_cost[0]),
                            total_bias=g.net_base[0] - g.net_base[L],
                            extra_oos_periods=g.oos_periods[L] - g.oos_periods[0],
                            extra_defects=g.defects[L] - g.defects[0]))
    pd.DataFrame(dec).to_csv(out / 'study2_bias_decomposition.csv', index=False)
    # 5) Chart operating characteristics (false-alarm and detection), from nominal/severe at delay 0
    oc = df[(df.delay == 0) & df.deployed.astype(bool)].groupby(['scenario', 'policy']).agg(
        rollback_rate=('rolled', 'mean'), n_deployed=('seed', 'size')).reset_index()
    oc.to_csv(out / 'study2_monitoring_oc.csv', index=False)
    return summ, pd.DataFrame(bias), pd.DataFrame(pdiff), agree


def acceptance_oc():
    from scipy.stats import binom
    return {f'{p:.3f}': float(binom.cdf(AQL_C, AQL_N, p)) for p in (.035, .06, .10, .12, .20, .24)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seeds', type=int, default=1000); a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    df = run(a.seeds); df.to_csv(a.output / 'study2_raw.csv.gz', index=False, compression='gzip')
    summ, bias, pdiff, agree = analyse(df, a.output)
    meta = dict(H=H, B=B, shift_period=SHIFT_T, scenarios=SCEN, delays=DELAYS, seeds=a.seeds,
                base_costs=dict(revenue_per_period=REVENUE, pilot=PILOT_COST, deploy=DEPLOY_COST,
                                defect_penalty=BASE_PEN, rollback=BASE_RB),
                cusum=dict(p0=P0, p1=P1, h=H_CUSUM), ewma=dict(lam=LAM, ucl=EWMA_UCL),
                acceptance_plan=dict(n=AQL_N, c=AQL_C, P_accept=acceptance_oc()),
                trajectories=len(df), ranking_agreement_rate=float(agree.same_best.mean()),
                max_regret=float(agree.regret.max()), mean_regret=float(agree.regret.mean()))
    (a.output / 'study2_meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
