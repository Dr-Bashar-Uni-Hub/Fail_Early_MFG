"""Delay-aware online slip calibration (Study 3).

Release rule: release an alarm if its (clipped) logit score s < theta. Feedback is action-dependent: a defect that is
inspected is confirmed at once (e = 0), whereas a released defect is discovered only after D days (e = 1), e.g. at
downstream test, field return or audit. Each confirmed defect updates the threshold as it arrives,
    theta <- theta + gamma * (alpha - e),
i.e. quantile tracking (Gibbs and Candes 2021; Angelopoulos, Candes and Tibshirani 2023) driven by delayed, action-dependent
feedback. Proposition 1 (manuscript): if scores lie in [-b, b] and at most M released defects await feedback at any time,
then after K confirmed defects
    | (1/K) sum_k e_k - alpha | <= (b + |theta_0| + gamma (M + 1)) / (gamma K),
deterministically, for any score sequence (no assumption on drift). The realized slip over all defects therefore converges to
alpha as K grows. The guarantee is on long-run average slip, not on any single week, and requires every released defect to
be discovered eventually.

Comparators on identical scores: the static 1%-slip threshold of the main analysis (refitted weekly) and the
confidence-bound rule's behaviour is reported from aoi_methods.py. Models are the weekly-refitted gradient-boosting models
of the main replay with model-refit delay equal to D.
"""
import argparse, heapq, json
from pathlib import Path
import numpy as np, pandas as pd
import aoi_replay as A

ALPHA = 0.01
B_CLIP = 12.0
GAMMAS = (0.02, 0.05, 0.1, 0.2)
DELAYS = (0, 1, 3, 7)


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return np.clip(np.log(p / (1 - p)), -B_CLIP + 1e-6, B_CLIP - 1e-6)


def online(scores, y, ts, D_days, gamma, theta0, alpha=ALPHA):
    """Process units in time order; return release flags, threshold path, feedback sequence, max in-flight count."""
    n = len(scores); rel = np.zeros(n, bool); theta = theta0; th = np.empty(n)
    due = []                      # (arrival_time, e) for released defects
    e_seq = []; inflight = 0; M = 0
    tsv = ts.to_numpy()
    for i in range(n):
        while due and due[0][0] <= tsv[i]:
            _, e = heapq.heappop(due); theta += gamma * (alpha - e); e_seq.append(e); inflight -= 1
        th[i] = theta; rel[i] = scores[i] < theta
        if y[i] == 1:
            if rel[i]:
                heapq.heappush(due, (tsv[i] + np.timedelta64(int(D_days * 86400), 's'), 1))
                inflight += 1; M = max(M, inflight)
            else:
                theta += gamma * (alpha - 0); e_seq.append(0)
    while due:
        _, e = heapq.heappop(due); e_seq.append(e)
    return rel, th, np.array(e_seq), M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(A.SEED); X, y, ts = A.load(a.data)
    start = int(A.BURN_IN * len(y)); rows, weekly = [], []
    # initial threshold: the static 1%-slip threshold from the burn-in history (same procedure as the main analysis)
    m0, thr0 = A.fit_and_threshold(X, y, np.arange(start), rng); theta0 = float(logit(np.array([thr0]))[0])
    for D in DELAYS:
        d = A.replay(X, y, ts, D, rng)          # weekly refits using labels released D days later; static threshold in rel_thr
        d = d.sort_values('row'); s = logit(d.score.to_numpy()); yy = d.y.to_numpy(); tt = ts.iloc[d.row.to_numpy()].reset_index(drop=True)
        variants = {'static 1% rule': d.rel_thr.to_numpy().astype(bool)}
        meta = {}
        for g in GAMMAS:
            rel, th, e, M = online(s, yy, tt, D, g, theta0)
            K = len(e); bound = (B_CLIP + abs(theta0) + g * (M + 1)) / (g * K)
            variants[f'online, gamma={g}'] = rel; meta[f'online, gamma={g}'] = dict(K=K, M=M, bound=bound, dev=abs(e.mean() - ALPHA))
        wk = d.week.to_numpy()
        for name, rel in variants.items():
            slip = rel[yy == 1].mean(); vr = rel[yy == 0].mean()
            ws = pd.Series(rel[yy == 1]).groupby(wk[yy == 1]).mean()
            rows.append(dict(delay_days=D, method=name, slip=slip, volume_reduction=vr, weeks_above=int((ws > ALPHA).sum()),
                             weeks=int(ws.size), max_weekly_slip=float(ws.max()), **meta.get(name, {})))
            for w_, v in ws.items():
                weekly.append(dict(delay_days=D, method=name, week=w_, slip=v))
        print('D =', D, 'done', flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.output / 'study3_online_calibration.csv', index=False)
    pd.DataFrame(weekly).to_csv(a.output / 'study3_online_weekly.csv', index=False)
    (a.output / 'study3_online_meta.json').write_text(json.dumps(dict(alpha=ALPHA, b_clip=B_CLIP, theta0_logit=theta0, gammas=GAMMAS, delays=DELAYS), indent=2))
    print(df.round(4).to_string())


if __name__ == '__main__':
    main()
