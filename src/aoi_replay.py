"""Study 3: Siemens AOI false-call data (Pfab et al. 2024, Data in Brief, doi:10.1016/j.dib.2024.110110;
Mendeley Data doi:10.17632/99jzmh9658.1). One row = one AOI-flagged solder joint; class 1 = true defect
(confirmed at manual inspection), 0 = false call. Timestamps are anonymized but order and spacing are kept.

Decision: release AOI alarms predicted to be false calls without manual inspection.
  (a) Threshold policy, as used in practice: on the most recent 20% of the labelled history, choose the
      highest score threshold that keeps slip (share of true defects released) <= 1%; apply it to new alarms.
      Outcomes: realized slip and volume reduction (share of false calls released).
  (b) Fixed release fraction: release the 40% lowest-risk alarms; outcome slip (random release: 0.40).
Designs: retrospective stratified random 60/40 split (repeated) and weekly deployment replay with label delay
L days, exactly as in Study 1. Uncertainty: calendar-day block bootstrap on day-level aggregates.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SEED = 20261123
RELEASE = 0.40
TARGET_SLIP = 0.01
DELAYS = [0, 1, 3, 7, 14]
BURN_IN = 0.40
NEG_SAMPLE = 0.20      # share of false calls kept for fitting (all true defects kept)
CAL_FRAC = 0.20        # most recent share of the labelled history used to set the threshold
CAT = ['inspection_type', 'meta_feat1', 'meta_feat2', 'meta_feat3', 'meta_feat4']


def load(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith('unnamed')])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)
    for c in CAT:
        df[c] = df[c].astype('category').cat.codes
    feats = CAT + [c for c in df.columns if c not in CAT + ['timestamp', 'class']]
    return df[feats], df['class'].astype(int).to_numpy(), df['timestamp']


def fit(X, y, idx, rng):
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    neg = np.sort(rng.choice(neg, max(1, int(NEG_SAMPLE * len(neg))), replace=False))
    t = np.concatenate([pos, neg])
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42,
                                       categorical_features=list(range(len(CAT))))
    m.fit(X.iloc[t], y[t]); return m


def threshold_for_slip(score, y, target=TARGET_SLIP):
    """Largest threshold t such that the share of true defects with score < t is <= target."""
    d = np.sort(score[y == 1]); k = int(np.floor(target * len(d)))
    return d[k] if len(d) else -np.inf      # releasing score < d[k] releases at most k defects


def fit_and_threshold(X, y, hist, rng, time_ordered=True):
    """Fit on the older part of the history; set the threshold on the most recent CAL_FRAC (or a random part)."""
    n_cal = max(1, int(CAL_FRAC * len(hist)))
    if time_ordered:
        tr, cal = hist[:-n_cal], hist[-n_cal:]
    else:
        tr, cal = train_test_split(hist, test_size=CAL_FRAC, stratify=y[hist], random_state=int(rng.integers(1e9)))
    m = fit(X, y, tr, rng)
    return m, threshold_for_slip(m.predict_proba(X.iloc[cal])[:, 1], y[cal])


def unit_outcomes(y, score, thr):
    rel_thr = score < thr
    k = int(round(RELEASE * len(y))); rel_fix = np.zeros(len(y), bool); rel_fix[np.argsort(score, kind='stable')[:k]] = True
    return rel_thr, rel_fix


def retrospective(X, y, repeats, rng):
    out = []
    for r in range(repeats):
        tr, te = train_test_split(np.arange(len(y)), test_size=.4, stratify=y, random_state=SEED + r)
        m, thr = fit_and_threshold(X, y, tr, rng, time_ordered=False)
        p = m.predict_proba(X.iloc[te])[:, 1]; rt, rf = unit_outcomes(y[te], p, thr); yt = y[te]
        out.append(dict(rep=r, slip_thr=rt[yt == 1].mean(), vr_thr=rt[yt == 0].mean(),
                        slip_fixed=rf[yt == 1].mean(), auc=roc_auc_score(yt, p)))
    return pd.DataFrame(out)


def replay(X, y, ts, L, rng, frozen=False):
    start = int(BURN_IN * len(y)); t0 = ts.iloc[start].normalize()
    edges = pd.date_range(t0, ts.iloc[-1] + pd.Timedelta(days=7), freq='7D'); rows = []; model = None
    for a, b in zip(edges[:-1], edges[1:]):
        win = np.where((ts >= a) & (ts < b) & (np.arange(len(y)) >= start))[0]
        if len(win) == 0:
            continue
        hist = np.arange(start) if frozen else np.where(ts + pd.Timedelta(days=L) <= a)[0]
        hist = hist[hist < win[0]]
        if model is None or not frozen:
            model = fit_and_threshold(X, y, hist, rng)
        m, thr = model; p = m.predict_proba(X.iloc[win])[:, 1]; rt, rf = unit_outcomes(y[win], p, thr)
        rows.append(pd.DataFrame(dict(row=win, day=ts.iloc[win].dt.strftime('%Y-%m-%d').values, week=str(a.date()),
                                      y=y[win], score=p, rel_thr=rt.astype(int), rel_fix=rf.astype(int),
                                      hist_n=len(hist), hist_defects=int(y[hist].sum()))))
    return pd.concat(rows, ignore_index=True)


def day_table(d):
    g = d.assign(d_rt=d.y * d.rel_thr, f_rt=(1 - d.y) * d.rel_thr, d_rf=d.y * d.rel_fix).groupby('day')
    return g.agg(n=('y', 'size'), D=('y', 'sum'), d_rt=('d_rt', 'sum'), f_rt=('f_rt', 'sum'), d_rf=('d_rf', 'sum'))


def ratios(t):
    D, F = t.D.sum(), (t.n - t.D).sum()
    return np.array([t.d_rt.sum() / D, t.f_rt.sum() / F, t.d_rf.sum() / D])


def boot(tables, rng, B):
    """Day-block bootstrap; the same resampled days are used for every table (paired)."""
    days = tables[0].index.values; arrs = [t.loc[days, ['n', 'D', 'd_rt', 'f_rt', 'd_rf']].to_numpy(float) for t in tables]
    out = []
    for _ in range(B):
        i = rng.integers(0, len(days), len(days))
        out.append([np.array([a[i, 2].sum() / a[i, 1].sum(), a[i, 3].sum() / (a[i, 0] - a[i, 1]).sum(), a[i, 4].sum() / a[i, 1].sum()]) for a in arrs])
    return np.array(out)       # B x tables x 3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--repeats', type=int, default=10); a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED); X, y, ts = load(a.data)
    retro = retrospective(X, y, a.repeats, rng); retro.to_csv(a.output / 'study3_retrospective.csv', index=False)
    print('retrospective done', flush=True)
    preds = {}
    for L in DELAYS:
        preds[L] = replay(X, y, ts, L, rng); print('replay L =', L, 'done', flush=True)
    preds['frozen'] = replay(X, y, ts, 0, rng, frozen=True)
    keys = list(preds); tabs = [day_table(preds[k]) for k in keys]
    days = sorted(set.intersection(*[set(t.index) for t in tabs])); tabs = [t.loc[days] for t in tabs]
    bs = boot(tabs, np.random.default_rng(SEED), a.boot)
    rows = []
    for j, k in enumerate(keys):
        est = ratios(tabs[j]); lo, hi = np.percentile(bs[:, j, :], [2.5, 97.5], axis=0); d = preds[k]
        wk = d[d.y == 1].groupby('week').rel_thr.mean()
        rows.append(dict(design='frozen' if k == 'frozen' else 'replay', delay_days=None if k == 'frozen' else k,
                         n=len(d), defects=int(d.y.sum()), weeks=int(d.week.nunique()),
                         slip_thr=est[0], slip_thr_lo=lo[0], slip_thr_hi=hi[0], vr_thr=est[1], vr_thr_lo=lo[1], vr_thr_hi=hi[1],
                         slip_fixed=est[2], slip_fixed_lo=lo[2], slip_fixed_hi=hi[2], auc=roc_auc_score(d.y, d.score),
                         weeks_slip_above_target=int((wk > TARGET_SLIP).sum()), max_weekly_slip=float(wk.max())))
    summ = pd.DataFrame(rows); summ.to_csv(a.output / 'study3_aoi_summary.csv', index=False)
    i0 = keys.index(0); contr = []
    for L in DELAYS[1:]:
        j = keys.index(L); diff = bs[:, j, :] - bs[:, i0, :]; est = ratios(tabs[j]) - ratios(tabs[i0])
        for q, nm in enumerate(['slip_thr', 'vr_thr', 'slip_fixed']):
            contr.append(dict(contrast=f'L={L} minus L=0', metric=nm, estimate=est[q],
                              lo=np.percentile(diff[:, q], 2.5), hi=np.percentile(diff[:, q], 97.5)))
    pd.DataFrame(contr).to_csv(a.output / 'study3_paired_delay_contrast.csv', index=False)
    wrows = []
    for k in keys:
        for wkey, g in preds[k].groupby('week'):
            wrows.append(dict(design=str(k), week=wkey, n=len(g), defects=int(g.y.sum()),
                              slip_thr=g.loc[g.y == 1, 'rel_thr'].mean() if g.y.sum() else np.nan,
                              vr_thr=g.loc[g.y == 0, 'rel_thr'].mean()))
    pd.DataFrame(wrows).to_csv(a.output / 'study3_weekly.csv', index=False)
    meta = dict(n=len(y), defects=int(y.sum()), defect_rate=float(y.mean()), first=str(ts.min()), last=str(ts.max()),
                deployment_start=str(ts.iloc[int(BURN_IN * len(y))]), release_fraction=RELEASE, target_slip=TARGET_SLIP,
                burn_in_fraction=BURN_IN, calibration_fraction=CAL_FRAC, negative_subsample=NEG_SAMPLE,
                bootstrap=f'calendar-day block bootstrap on day aggregates, {a.boot} resamples, {len(days)} days',
                retrospective={c: dict(mean=float(retro[c].mean()), p2_5=float(retro[c].quantile(.025)), p97_5=float(retro[c].quantile(.975)))
                               for c in ('slip_thr', 'vr_thr', 'slip_fixed', 'auc')}, repeats=a.repeats)
    (a.output / 'study3_meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta['retrospective'], indent=2)); print(summ.round(4).to_string()); print(pd.DataFrame(contr).round(4).to_string())


if __name__ == '__main__':
    main()
