"""Study 3 robustness: do other models or better threshold strategies close the gap between the random-split
estimate and deployment on the Siemens AOI data? Weekly replay with immediate labels (L = 0).

Models (default 1%-slip threshold on the most recent 20% of history):
  hgb            histogram gradient boosting (main text)
  rf             random forest, class-weighted (balanced_subsample), 300 trees
  logit          L2 logistic regression on standardized numeric + one-hot categorical features
Threshold strategies (hgb scores):
  default        1% slip on the most recent 20% of labelled history
  recent_week    1% slip on the last 7 days of labelled history only
  target_0.5     0.5% slip target on the most recent 20%
  ucb            largest threshold whose one-sided 95% Clopper-Pearson upper bound on slip is <= 1%
Each model also gets a retrospective random-split estimate (3 repeats) for its own gap.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import beta
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import aoi_replay as A

TARGET = 0.01


def make(name, X):
    cat = list(range(len(A.CAT))); num = list(range(len(A.CAT), X.shape[1]))
    if name == 'hgb':
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=.1, random_state=42, categorical_features=cat)
    if name == 'rf':
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight='balanced_subsample',
                                      max_features='sqrt', n_jobs=-1, random_state=42)
    pre = ColumnTransformer([('c', OneHotEncoder(handle_unknown='ignore'), cat), ('n', StandardScaler(), num)])
    return Pipeline([('p', pre), ('m', LogisticRegression(C=.1, max_iter=2000))])


def fit(name, X, y, idx, rng):
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    neg = np.sort(rng.choice(neg, max(1, int(A.NEG_SAMPLE * len(neg))), replace=False))
    t = np.concatenate([pos, neg]); m = make(name, X); m.fit(X.iloc[t].to_numpy(), y[t]); return m


def thr_point(score, y, target):
    return A.threshold_for_slip(score, y, target)


def thr_ucb(score, y, target=TARGET, conf=.95):
    d = np.sort(score[y == 1]); n = len(d); best = -np.inf
    for k in range(n):
        ub = beta.ppf(conf, k + 1, n - k)       # one-sided Clopper-Pearson upper bound for k released of n
        if ub > target:
            break
        best = d[k]                              # release score < d[k] -> k defects released
    return best


def week_rows(win, ts, y, p, thr, label, week):
    rel = p < thr
    k = int(round(A.RELEASE * len(win))); fix = np.zeros(len(win), bool); fix[np.argsort(p, kind='stable')[:k]] = True
    return pd.DataFrame(dict(variant=label, row=win, day=ts.iloc[win].dt.strftime('%Y-%m-%d').values, y=y[win],
                             score=p, rel_thr=rel.astype(int), rel_fix=fix.astype(int), week=week))


def replay(X, y, ts, rng):
    start = int(A.BURN_IN * len(y)); t0 = ts.iloc[start].normalize()
    edges = pd.date_range(t0, ts.iloc[-1] + pd.Timedelta(days=7), freq='7D'); out = []
    for a, b in zip(edges[:-1], edges[1:]):
        win = np.where((ts >= a) & (ts < b) & (np.arange(len(y)) >= start))[0]
        if len(win) == 0:
            continue
        hist = np.where(ts <= a)[0]; hist = hist[hist < win[0]]
        n_cal = int(A.CAL_FRAC * len(hist)); tr, cal = hist[:-n_cal], hist[-n_cal:]
        wk = cal[ts.iloc[cal].to_numpy() >= (a - pd.Timedelta(days=7))]
        Xw = X.iloc[win].to_numpy()
        for name in ('hgb', 'rf', 'logit'):
            m = fit(name, X, y, tr, rng); sc = m.predict_proba(X.iloc[cal].to_numpy())[:, 1]; p = m.predict_proba(Xw)[:, 1]
            out.append(week_rows(win, ts, y, p, thr_point(sc, y[cal], TARGET), f'model_{name}', str(a.date())))
            if name == 'hgb':
                scw = sc[np.isin(cal, wk)]
                out.append(week_rows(win, ts, y, p, thr_point(scw, y[wk], TARGET), 'thr_recent_week', str(a.date())))
                out.append(week_rows(win, ts, y, p, thr_point(sc, y[cal], .005), 'thr_target_0.5', str(a.date())))
                out.append(week_rows(win, ts, y, p, thr_ucb(sc, y[cal]), 'thr_ucb', str(a.date())))
        print('week', a.date(), 'done', flush=True)
    return pd.concat(out, ignore_index=True)


def retro(X, y, rng, repeats):
    rows = []
    for name in ('hgb', 'rf', 'logit'):
        for r in range(repeats):
            tr, te = train_test_split(np.arange(len(y)), test_size=.4, stratify=y, random_state=A.SEED + r)
            fi, ca = train_test_split(tr, test_size=A.CAL_FRAC, stratify=y[tr], random_state=A.SEED + 100 + r)
            m = fit(name, X, y, fi, rng); thr = thr_point(m.predict_proba(X.iloc[ca].to_numpy())[:, 1], y[ca], TARGET)
            p = m.predict_proba(X.iloc[te].to_numpy())[:, 1]; yt = y[te]
            k = int(round(A.RELEASE * len(te))); fix = np.zeros(len(te), bool); fix[np.argsort(p, kind='stable')[:k]] = True
            rows.append(dict(model=name, rep=r, slip_thr=(p[yt == 1] < thr).mean(), vr_thr=(p[yt == 0] < thr).mean(),
                             slip_fixed=fix[yt == 1].mean(), auc=roc_auc_score(yt, p)))
        print('retro', name, 'done', flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--repeats', type=int, default=3); a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(A.SEED); X, y, ts = A.load(a.data)
    rt = retro(X, y, rng, a.repeats); rt.to_csv(a.output / 'study3_methods_retrospective.csv', index=False)
    d = replay(X, y, ts, rng)
    variants = list(d.variant.unique()); tabs = {v: A.day_table(d[d.variant == v]) for v in variants}
    days = sorted(set.intersection(*[set(t.index) for t in tabs.values()])); tl = [tabs[v].loc[days] for v in variants]
    bs = A.boot(tl, np.random.default_rng(A.SEED), a.boot); rows = []
    for j, v in enumerate(variants):
        est = A.ratios(tl[j]); lo, hi = np.percentile(bs[:, j, :], [2.5, 97.5], axis=0); g = d[d.variant == v]
        wk = g[g.y == 1].groupby('week').rel_thr.mean()
        rr = rt[rt.model == v.replace('model_', '')] if v.startswith('model_') else None
        rows.append(dict(variant=v, slip_thr=est[0], slip_thr_lo=lo[0], slip_thr_hi=hi[0], vr_thr=est[1], vr_thr_lo=lo[1],
                         vr_thr_hi=hi[1], slip_fixed=est[2], slip_fixed_lo=lo[2], slip_fixed_hi=hi[2],
                         auc=roc_auc_score(g.y, g.score), weeks_above=int((wk > TARGET).sum()), weeks=int(wk.size),
                         retro_slip_thr=None if rr is None else rr.slip_thr.mean(), retro_vr_thr=None if rr is None else rr.vr_thr.mean(),
                         retro_slip_fixed=None if rr is None else rr.slip_fixed.mean(), retro_auc=None if rr is None else rr.auc.mean()))
    pd.DataFrame(rows).to_csv(a.output / 'study3_methods_summary.csv', index=False)
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == '__main__':
    main()
