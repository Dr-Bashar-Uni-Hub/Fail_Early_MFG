"""Latency requirement for a quality-exposure limit (Study 2, severe shift).

Proposition 2 (manuscript): if a policy is deployed before the shift and its monitoring statistic sees the same post-shift
batch outcomes whatever the delay (common random numbers, rule unchanged), a report delay of L periods postpones rollback by
exactly L periods, so the out-of-specification exposure of a run is min(H_post, Z + L), where Z is its exposure with
immediate results and H_post the number of post-shift periods. Hence
    E(L) = E[ D * min(H_post, Z + L) ],   E(L+1) - E(L) = P(D = 1, Z + L < H_post) <= 1,
with D the deployment indicator, and the maximum admissible delay for an exposure limit B is
    L_max(B) = max{ L : E(L) <= B }.
This script predicts E(L) from the L = 0 runs only and compares it with the simulated exposure at L = 1..4 for every policy;
policies whose rules violate the assumptions (WS uses cumulative rates that include a delay-dependent amount of pre-shift
data) serve as counterexamples.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import delay_sim as S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', type=Path, required=True); a = ap.parse_args()
    raw = pd.read_csv(a.results / 'study2_raw.csv.gz'); raw = raw[raw.scenario == 'severe_shift']
    Hp = S.H - S.SHIFT_T; rows = []
    for pol, g in raw.groupby('policy'):
        g0 = g[g.delay == 0].set_index('seed'); Z, D = g0.oos_periods.to_numpy(), g0.deployed.to_numpy()
        for L in S.DELAYS:
            pred = np.mean(D * np.minimum(Hp, Z + L)); sim = g[g.delay == L].oos_periods.mean()
            per_run_exact = np.mean(D * np.minimum(Hp, Z + L) == g[g.delay == L].set_index('seed').loc[g0.index].oos_periods.to_numpy())
            rows.append(dict(policy=pol, delay=L, predicted=pred, simulated=sim, abs_error=abs(pred - sim),
                             share_runs_exact=per_run_exact, slope_pred=np.mean(D * ((Z + L) < Hp))))
    df = pd.DataFrame(rows); df.to_csv(a.results / 'study2_exposure_latency.csv', index=False)
    ac = df[df.policy == 'AC'].set_index('delay')
    lmax = {}
    for B in (1.0, 2.0, 3.0, 4.0, 6.0):
        ok_p = [L for L in S.DELAYS if ac.predicted[L] <= B]; ok_s = [L for L in S.DELAYS if ac.simulated[L] <= B]
        lmax[str(B)] = dict(predicted=max(ok_p) if ok_p else None, simulated=max(ok_s) if ok_s else None)
    (a.results / 'study2_latency_requirement.json').write_text(json.dumps(dict(policy='AC', scenario='severe_shift', H_post=Hp, L_max=lmax), indent=2))
    print(df.round(3).to_string()); print(json.dumps(lmax))


if __name__ == '__main__':
    main()
