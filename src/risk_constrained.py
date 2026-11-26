"""Risk-constrained policy choice for Study 2: maximize mean net value subject to mean out-of-specification
periods per run <= q_max, compared with the unconstrained (value-only) choice, for every scenario, cost pair and delay."""
import argparse
from pathlib import Path
import pandas as pd
import delay_sim as S

QMAX = (1.0, 2.0, 4.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', type=Path, required=True); a = ap.parse_args()
    raw = pd.read_csv(a.results / 'study2_raw.csv.gz'); rows = []
    for pen in S.PENALTIES:
        for rb in S.ROLLBACKS:
            v = raw.assign(net=S.net_value(raw, pen, rb)).groupby(['scenario', 'delay', 'policy']).agg(
                net=('net', 'mean'), oos=('oos_periods', 'mean')).reset_index()
            for (sc, L), g in v.groupby(['scenario', 'delay']):
                best = g.loc[g.net.idxmax()]
                for q in QMAX:
                    feas = g[g.oos <= q]
                    pick = feas.loc[feas.net.idxmax()] if len(feas) else None
                    rows.append(dict(scenario=sc, delay=L, penalty=pen, rollback_cost=rb, q_max=q,
                                     best_value_only=best.policy, net_value_only=best.net, oos_value_only=best.oos,
                                     best_constrained=None if pick is None else pick.policy,
                                     net_constrained=None if pick is None else pick.net,
                                     oos_constrained=None if pick is None else pick.oos,
                                     changed=(pick is not None and pick.policy != best.policy)))
    df = pd.DataFrame(rows); df.to_csv(a.results / 'study2_risk_constrained.csv', index=False)
    ch = df[df.changed]
    print(df.groupby(['scenario', 'q_max']).changed.mean().round(3).to_string())
    print(ch.assign(value_cost=ch.net_value_only - ch.net_constrained, oos_saved=ch.oos_value_only - ch.oos_constrained)
          .groupby(['scenario', 'q_max'])[['value_cost', 'oos_saved']].mean().round(2).to_string())


if __name__ == '__main__':
    main()
