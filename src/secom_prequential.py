"""Study 1: prequential (deployment-replay) evaluation of SECOM screening under label delay.

Records are processed in timestamp order. At the start of each weekly window the model is
refit using only records whose quality label would have been released (timestamp + L <= window
start). The window is then scored and the top-q fraction (inspection budget) is flagged for
extra inspection. This is compared with the retrospective random-split estimate an analyst
would usually report. Uncertainty: day-block bootstrap (resampling calendar days).
"""
import argparse, io, zipfile, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 20261123
BUDGET = 0.10          # inspect 10% of units
DELAYS = [0, 1, 3, 7, 14]   # label delay in days
BURN_IN = 0.40         # first 40% of records form the initial labelled history


def models():
    return {
        'logistic': Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler()),
                              ('c', LogisticRegression(C=.1, max_iter=3000, random_state=42))]),
        'random_forest': Pipeline([('i', SimpleImputer(strategy='median')),
                                   ('c', RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                                                n_jobs=-1, random_state=42))]),
    }


def keep_cols(X):
    return X.columns[X.notna().any() & (X.nunique(dropna=True) > 1)]


def flag_top(scores, budget):
    k = max(1, int(round(budget * len(scores))))
    idx = np.argsort(-scores, kind='stable')[:k]
    f = np.zeros(len(scores), bool); f[idx] = True
    return f


def load(archive):
    with zipfile.ZipFile(archive) as z:
        X = pd.read_csv(io.BytesIO(z.read('secom.data')), sep=r'\s+', header=None, na_values=['NaN'])
        lab = pd.read_csv(io.BytesIO(z.read('secom_labels.data')), sep=r'\s+', header=None)
    y = (lab.iloc[:, 0] == 1).astype(int).to_numpy()
    ts = pd.to_datetime(lab.iloc[:, 1], format='%d/%m/%Y %H:%M:%S')
    o = np.argsort(ts.to_numpy(), kind='stable')
    return X.iloc[o].reset_index(drop=True), y[o], ts.iloc[o].reset_index(drop=True)


def retrospective(X, y, repeats=50):
    """What an analyst usually reports: stratified random 60/40 splits (repeated), top-10% recall on the test part."""
    rec = {k: [] for k in models()}
    for r in range(repeats):
        tr, te = train_test_split(np.arange(len(y)), test_size=.4, stratify=y, random_state=SEED + r)
        cols = keep_cols(X.iloc[tr])
        for name, m in models().items():
            m.fit(X.iloc[tr][cols], y[tr]); p = m.predict_proba(X.iloc[te][cols])[:, 1]
            f = flag_top(p, BUDGET)
            rec[name].append(dict(rep=r, model=name, roc_auc=roc_auc_score(y[te], p),
                                  pr_auc=average_precision_score(y[te], p),
                                  recall_at_budget=y[te][f].sum() / y[te].sum(),
                                  precision_at_budget=y[te][f].mean()))
    rows = [x for v in rec.values() for x in v]
    df = pd.DataFrame(rows)
    out = {}
    for name, d in df.groupby('model'):
        out[name] = {m: dict(mean=float(d[m].mean()), p2_5=float(d[m].quantile(.025)), p97_5=float(d[m].quantile(.975)))
                     for m in ('roc_auc', 'pr_auc', 'recall_at_budget', 'precision_at_budget')}
    return out, df


def available_indices(ts, delay_days, window_start, first_window_row):
    """Records whose label is released by window_start (timestamp + delay <= window_start) and that precede the window."""
    avail = np.where(ts + pd.Timedelta(days=delay_days) <= window_start)[0]
    return avail[avail < first_window_row]


def prequential(X, y, ts, delay_days, frozen=False, window_days=7):
    start = int(BURN_IN * len(y))
    t0 = ts.iloc[start].normalize()
    edges = pd.date_range(t0, ts.iloc[-1] + pd.Timedelta(days=window_days), freq=f'{window_days}D')
    rows = []
    for name in ('logistic', 'random_forest'):
        fitted = None
        for a, b in zip(edges[:-1], edges[1:]):
            win = np.where((ts >= a) & (ts < b) & (np.arange(len(y)) >= start))[0]
            if len(win) == 0:
                continue
            if frozen:
                avail = np.arange(start)
            else:
                avail = available_indices(ts, delay_days, a, win[0])
            if fitted is None or not frozen:
                cols = keep_cols(X.iloc[avail]); m = models()[name]
                m.fit(X.iloc[avail][cols], y[avail]); fitted = (m, cols)
            m, cols = fitted
            p = m.predict_proba(X.iloc[win][cols])[:, 1]
            f = flag_top(p, BUDGET)
            for i, s, fl in zip(win, p, f):
                rows.append(dict(model=name, row=int(i), day=str(ts.iloc[i].date()), week=str(a.date()),
                                 y=int(y[i]), score=float(s), flagged=int(fl), train_n=len(avail),
                                 train_fail=int(y[avail].sum())))
    return pd.DataFrame(rows)


def summarise(df, rng, B=2000):
    """Point estimates + day-block bootstrap CIs. Scores are only compared within week (flags are per week)."""
    res = []
    for name, d in df.groupby('model'):
        days = d['day'].unique(); groups = {k: g for k, g in d.groupby('day')}
        def stats(g):
            rec = g.loc[g.y == 1, 'flagged'].mean() if g.y.sum() else np.nan
            prec = g.loc[g.flagged == 1, 'y'].mean()
            auc = roc_auc_score(g.y, g.score) if 0 < g.y.sum() < len(g) else np.nan
            return rec, prec, auc
        r0, p0, a0 = stats(d)
        bs = []
        for _ in range(B):
            g = pd.concat([groups[k] for k in rng.choice(days, len(days), replace=True)])
            bs.append(stats(g))
        bs = np.array(bs, float)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        res.append(dict(model=name, n=len(d), failures=int(d.y.sum()), flagged=int(d.flagged.sum()),
                        recall_at_budget=r0, recall_lo=lo[0], recall_hi=hi[0],
                        precision_at_budget=p0, precision_lo=lo[1], precision_hi=hi[1],
                        pooled_roc_auc=a0, auc_lo=lo[2], auc_hi=hi[2],
                        median_train_n=float(d.groupby('week').train_n.first().median()),
                        min_train_fail=int(d.train_fail.min())))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--secom-zip', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    X, y, ts = load(a.secom_zip)
    retro, retro_df = retrospective(X, y)
    retro_df.to_csv(a.output / 'study1_retrospective_repeats.csv', index=False)
    out, preds = [], []
    for L in DELAYS:
        df = prequential(X, y, ts, L); df['delay_days'] = L; preds.append(df)
        for r in summarise(df, rng, a.boot):
            out.append(dict(design='prequential', delay_days=L, **r))
    df = prequential(X, y, ts, 0, frozen=True); df['delay_days'] = -1; preds.append(df)
    for r in summarise(df, rng, a.boot):
        out.append(dict(design='frozen_burn_in_model', delay_days=np.nan, **r))
    pd.DataFrame(out).to_csv(a.output / 'study1_prequential_summary.csv', index=False)
    pd.concat(preds).to_csv(a.output / 'study1_prequential_predictions.csv', index=False)
    meta = dict(budget=BUDGET, burn_in_fraction=BURN_IN, delays_days=DELAYS, window='7 days',
                retrospective_random_split=retro, timestamp_range=[str(ts.min()), str(ts.max())],
                deployment_start=str(ts.iloc[int(BURN_IN * len(y))]), random_inspection_recall=BUDGET,
                bootstrap='calendar-day block bootstrap, %d resamples, seed %d' % (a.boot, SEED))
    (a.output / 'study1_meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(retro, indent=2)); print(pd.DataFrame(out).round(3).to_string())


if __name__ == '__main__':
    main()
