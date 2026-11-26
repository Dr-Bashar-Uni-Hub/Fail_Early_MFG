"""Study 3 extensions (V6.1): selective feedback with randomized post-release audits, additional calibration baselines on
identical scores, and finite-horizon slip with pending confirmations.

Feedback modes
  full      : every released defect is discovered D days later (assumption of Proposition 1).
  audit q   : released alarms are never labelled unless audited; each released alarm is audited independently with
              probability q, with the result available one day later. Inspected defects are confirmed at once.
Online update with inverse-propensity weights (Proposition 3): inspected defect theta += gamma*alpha;
audited released defect theta += gamma*(alpha - 1)/q.
Baselines on identical scores (full feedback, discovery delay D): static weekly 1%-slip rule; sliding-window quantile of
confirmed defect scores (28 days); exponentially weighted quantile (half-life 14 days); Clopper-Pearson window rule
(largest threshold whose one-sided 95% upper bound on slip among confirmed defects in the last 28 days is <= 1%).
Finite horizon (Corollary 1): at each week end, actual slip over all defects to date versus the observable bound
B_conf(C_T) + P_hat/(C_T + P_hat), where P_hat = released alarms in the last D days (>= pending released defects).
"""
import argparse, heapq, json, math
from collections import deque
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import beta
import aoi_replay as A
import aoi_online as O

ALPHA, GAMMA, AUDIT_DELAY = 0.01, 0.1, 1.0
QS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)


def online_feedback(s, y, t, gamma, theta0, D=None, q=0.0, rng=None, alpha=ALPHA, track=None):
    """t in days (float). D: discovery delay for unaudited released defects (None = never). Returns release, audited, M."""
    n = len(s); rel = np.zeros(n, bool); aud = np.zeros(n, bool); th = theta0; heap = []; pend = 0; M = 0
    for i in range(n):
        while heap and heap[0][0] <= t[i]:
            _, upd = heapq.heappop(heap); th += gamma * upd; pend -= 1
        rel[i] = s[i] < th
        if rel[i]:
            if q > 0 and rng.random() < q:
                aud[i] = True
                if y[i]:
                    heapq.heappush(heap, (t[i] + AUDIT_DELAY, (alpha - 1) / q)); pend += 1
            elif D is not None and y[i]:
                heapq.heappush(heap, (t[i] + D, alpha - 1)); pend += 1
            M = max(M, pend)
        elif y[i]:
            th += gamma * alpha
    return rel, aud, M


def window_rule(s, y, t, D, kind, win=28.0, halflife=14.0, alpha=ALPHA, fallback=None):
    """Threshold recomputed daily from defects confirmed by then (inspected at once, released after D days)."""
    n = len(s); rel = np.zeros(n, bool); conf = []      # heap of (confirm_time, score, time)
    buf = deque(); day = None; thr = fallback
    for i in range(n):
        d = math.floor(t[i])
        if d != day:
            day = d
            while conf and conf[0][0] <= t[i]:
                c = heapq.heappop(conf); buf.append((c[2], c[1]))
            while buf and buf[0][0] < t[i] - (win if kind != 'ew' else 5 * halflife):
                buf.popleft()
            sc = np.array([b[1] for b in buf])
            if len(sc) >= 30:
                if kind == 'window':
                    thr = np.quantile(sc, alpha, method='lower')
                elif kind == 'ew':
                    w = np.exp(-(t[i] - np.array([b[0] for b in buf])) * math.log(2) / halflife); o = np.argsort(sc)
                    cw = np.cumsum(w[o]) / w.sum(); thr = sc[o][np.searchsorted(cw, alpha)]
                else:                                    # Clopper-Pearson window rule
                    srt = np.sort(sc); m = len(srt); kmax = -1
                    for k in range(m):
                        if beta.ppf(.95, k + 1, m - k) > alpha:
                            break
                        kmax = k
                    thr = srt[kmax] if kmax >= 0 else -np.inf
        rel[i] = s[i] < thr
        if y[i]:
            heapq.heappush(conf, ((t[i] + D) if rel[i] else t[i], s[i], t[i]))
    return rel


def summary(name, rel, aud, y, wk, extra=None):
    slip = rel[y == 1].mean(); vr = (rel & ~aud)[y == 0].mean()
    ws = pd.Series(rel[y == 1]).groupby(wk[y == 1]).mean()
    return dict(method=name, slip=slip, volume_reduction_net=vr, audit_share_of_alarms=aud.mean(),
                weeks_above=int((ws > ALPHA).sum()), weeks=int(ws.size), max_weekly_slip=float(ws.max()), **(extra or {}))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--D', type=float, default=7.0)
    ap.add_argument('--audit-seeds', type=int, default=5); a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(A.SEED); X, y, ts = A.load(a.data); start = int(A.BURN_IN * len(y))
    m0, thr0 = A.fit_and_threshold(X, y, np.arange(start), rng); theta0 = float(O.logit(np.array([thr0]))[0])
    d = A.replay(X, y, ts, int(a.D), rng).sort_values('row')
    s = O.logit(d.score.to_numpy()); yy = d.y.to_numpy(); tt = ts.iloc[d.row.to_numpy()]
    tday = ((tt - tt.iloc[0]).dt.total_seconds() / 86400).to_numpy(); wk = d.week.to_numpy(); rows = []
    rows.append(summary('static 1% rule (full feedback)', d.rel_thr.to_numpy().astype(bool), np.zeros(len(s), bool), yy, wk))
    rel, aud, M = online_feedback(s, yy, tday, GAMMA, theta0, D=a.D); rows.append(summary('online (full feedback)', rel, aud, yy, wk, dict(M=M)))
    rel_full = rel
    for kind, nm in (('window', 'sliding-window quantile (28 d)'), ('ew', 'exponentially weighted quantile (14 d)'), ('cp', 'Clopper-Pearson window rule (28 d)')):
        r = window_rule(s, yy, tday, a.D, kind, fallback=theta0); rows.append(summary(nm + ' (full feedback)', r, np.zeros(len(s), bool), yy, wk))
    # selective feedback with audits (several audit randomizations)
    N = int(yy.sum()); lg = math.log(2 / 0.05)
    for q in QS:
        for sd in range(a.audit_seeds if 0 < q < 1 else 1):
            r, au, M = online_feedback(s, yy, tday, GAMMA, theta0, D=None, q=q, rng=np.random.default_rng(1000 + sd))
            if q > 0:
                det = (O.B_CLIP + abs(theta0) + GAMMA * max(ALPHA, (1 - ALPHA) * M / q)) / (GAMMA * N)
                mg = (1 - ALPHA) * (2 * lg / (3 * q) + math.sqrt(2 * N * (1 - q) / q * lg)) / N
                extra = dict(q=q, audit_seed=sd, M=M, bound_det=det, bound_mg=mg, bound=det + mg)
            else:
                extra = dict(q=q, audit_seed=sd, M=M, bound=None)
            row = summary(f'online, audit q={q} (no other feedback)', r, au, yy, wk, extra); row['dev'] = abs(row['slip'] - ALPHA); rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(a.output / 'study3_selective_baselines.csv', index=False)
    # finite-horizon analysis for online (full feedback)
    fh = []; ws_ends = pd.Series(tday).groupby(wk).max()
    for w_, tend in ws_ends.items():
        idx = tday <= tend; yi = yy[idx]; ri = rel_full[idx]; ti = tday[idx]
        Nt = yi.sum(); St = (ri & (yi == 1)).sum()
        confirmed = (yi == 1) & (~ri | (ti + a.D <= tend)); Ct = confirmed.sum(); Pt = Nt - Ct
        Sconf = (ri & confirmed).sum(); Phat = (ri & (ti > tend - a.D)).sum()
        Minf = Pt  # pending at T is <= M; conservative bound uses observable Phat
        bconf = (O.B_CLIP + abs(theta0) + GAMMA * max(ALPHA, (1 - ALPHA) * Phat)) / (GAMMA * max(Ct, 1))
        fh.append(dict(week=w_, N=int(Nt), C=int(Ct), P=int(Pt), P_hat=int(Phat), R=St / Nt, R_conf=Sconf / max(Ct, 1),
                       dev=abs(St / Nt - ALPHA), pending_gap=abs(St / Nt - Sconf / max(Ct, 1)), pending_bound=Phat / (Ct + Phat) if Ct + Phat else np.nan,
                       total_bound=bconf + (Phat / (Ct + Phat) if Ct + Phat else 0)))
    pd.DataFrame(fh).to_csv(a.output / 'study3_finite_horizon.csv', index=False)
    (a.output / 'study3_selective_meta.json').write_text(json.dumps(dict(D=a.D, gamma=GAMMA, alpha=ALPHA, audit_delay_days=AUDIT_DELAY, defects=N, theta0=theta0), indent=2))
    print(df.round(4).to_string()); print(pd.DataFrame(fh).round(4).to_string())


if __name__ == '__main__':
    main()
