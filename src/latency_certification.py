"""Statistically certified reporting-latency requirement (severe shift).

From n immediate-feedback runs, certify the largest delay L in a prespecified set such that the expected exposure E(L)
is at most B with probability >= 1 - delta, simultaneously over the set.

Under Proposition 2, run i's exposure at delay L is g_L(U_i) with U_i = -1 if not deployed, else Z_i, and
g_L(u) = 0 if u = -1, min(H, u + L) otherwise; each g_L is nondecreasing in u and bounded by H. A one-sided
Dvoretzky-Kiefer-Wolfowitz band (Massart 1990), F >= F_hat - eps with eps = sqrt(log(1/delta) / (2n)), holds with
probability >= 1 - delta and implies, for every nondecreasing g bounded by H, E[g(U)] <= mean of g after moving mass eps from
the smallest observations to the value H. The same band serves every L, so no multiplicity correction is needed.
Comparators: plug-in, Bonferroni-Hoeffding and Bonferroni empirical-Bernstein (Maurer and Pontil 2009) upper bounds.
Truth: E(L) estimated by direct simulation at each L from a large reference population (which also reveals where the
Proposition 2 assumptions fail, e.g. for EWMA).
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import delay_sim as S

DELAYS = (0, 1, 2, 3, 4)


def reference(policy, n_ref):
    pa, pb = S.SCEN['severe_shift']; p_path = np.array([pa if t < S.SHIFT_T else pb for t in range(S.H)]); rows = []
    for seed in range(n_ref):
        u = np.random.default_rng([99991, seed]).random((S.H, S.B))
        rec = {L: S.simulate(u, p_path, policy, L) for L in DELAYS}
        rows.append(dict(D=rec[0]['deployed'], Z=rec[0]['oos_periods'], **{f'X{L}': rec[L]['oos_periods'] for L in DELAYS}))
    return pd.DataFrame(rows)


def g(Ui, L, Hp):
    return np.where(Ui < 0, 0.0, np.minimum(Hp, Ui + L))


def bounds(Ui, Hp, delta):
    n = len(Ui); out = {}
    eps = np.sqrt(np.log(1 / delta) / (2 * n)); k = len(DELAYS); srt = np.sort(Ui)
    for L in DELAYS:
        x = g(srt, L, Hp); m = x.mean()
        # DKW worst case: remove mass eps from the smallest values, add it at value H
        w = np.full(n, 1 / n); rem = eps; i = 0
        while rem > 1e-15 and i < n:
            take = min(w[i], rem); w[i] -= take; rem -= take; i += 1
        dkw = float((w * x).sum() + min(eps, 1.0) * Hp)
        v = x.var(ddof=1) if n > 1 else 0.0; lg = np.log(2 * k / delta)
        out[L] = dict(plugin=m, hoeffding=m + Hp * np.sqrt(np.log(k / delta) / (2 * n)),
                      bernstein=m + np.sqrt(2 * v * lg / n) + 7 * Hp * lg / (3 * (n - 1)), dkw=min(dkw, Hp))
    return out


def lcert(ub, B):
    ok = [L for L in DELAYS if ub[L] <= B]
    return max(ok) if ok else -1          # -1: no delay certified (empty feasible set)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--n-ref', type=int, default=20000); ap.add_argument('--reps', type=int, default=1000); a = ap.parse_args()
    Hp = S.H - S.SHIFT_T; rng = np.random.default_rng(2026); rows = []; truth = {}
    for pol in ('AC', 'AE'):
        ref = reference(pol, a.n_ref); U = np.where(ref.D == 1, ref.Z, -1).astype(float)
        E_true = {L: ref[f'X{L}'].mean() for L in DELAYS}; E_pred = {L: g(U, L, Hp).mean() for L in DELAYS}; truth[pol] = dict(direct=E_true, prop2=E_pred)
        for B in (2.0, 3.0, 4.0):
            Lstar = max([L for L in DELAYS if E_true[L] <= B], default=-1)
            for n in (50, 100, 200, 500, 1000):
                for delta in (0.05,):
                    res = {m: dict(cov=0, fc=0, L=[]) for m in ('plugin', 'hoeffding', 'bernstein', 'dkw')}
                    for _ in range(a.reps):
                        idx = rng.integers(0, len(U), n); b = bounds(U[idx], Hp, delta)
                        for m in res:
                            ub = {L: b[L][m] for L in DELAYS}
                            res[m]['cov'] += all(E_true[L] <= ub[L] for L in DELAYS)
                            Lc = lcert(ub, B); res[m]['L'].append(Lc); res[m]['fc'] += Lc > Lstar
                    for m, r in res.items():
                        Ls = np.array(r['L'])
                        rows.append(dict(policy=pol, B=B, n=n, delta=delta, method=m, true_Lmax=Lstar,
                                         simultaneous_coverage=r['cov'] / a.reps, false_certification=r['fc'] / a.reps,
                                         mean_Lcert=float(Ls[Ls >= 0].mean()) if (Ls >= 0).any() else None,
                                         share_none_certified=float((Ls < 0).mean())))
        print(pol, 'done', flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.output / 'latency_certification.csv', index=False)
    (a.output / 'latency_certification_truth.json').write_text(json.dumps(truth, indent=2))
    print(df[(df.B == 3.0)].round(3).to_string())


if __name__ == '__main__':
    main()
