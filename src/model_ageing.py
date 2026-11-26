"""Model ageing: how quickly does a quality model go stale? Explains when label delay matters.

At weekly anchors a (from the deployment start), a model is fitted on all units with timestamp < a and applied to
units whose age, timestamp - a, falls in bands of 0-7, 7-14, 14-21, 21-28 and 28-42 days. Within each anchor x band
group the same decision as in the main analysis is applied (SECOM: flag the top 10%; AOI: release the 40% lowest
risk), and outcomes are pooled over anchors by band. A model that loses value quickly with age makes label delay
costly; a flat curve means delay matters little. Uncertainty: calendar-day block bootstrap within band.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import secom_prequential as P
import aoi_replay as A

BANDS = [(0, 7), (7, 14), (14, 21), (21, 28), (28, 42)]
SEED = 20261123


def anchors(ts, burn_in):
    t0 = ts.iloc[int(burn_in * len(ts))].normalize()
    return list(pd.date_range(t0, ts.iloc[-1] - pd.Timedelta(days=7), freq='7D'))


def run(ts, y, fit_score, budget, mode, rng_boot, B):
    """fit_score(train_idx, test_idx) -> scores. mode 'flag' (top budget, recall) or 'release' (bottom budget, slip)."""
    recs = []
    for a in anchors(ts, P.BURN_IN):
        tr = np.where(ts < a)[0]
        age = (ts - a).dt.total_seconds().to_numpy() / 86400
        te = np.where((age >= 0) & (age < BANDS[-1][1]))[0]
        if len(te) == 0:
            continue
        s = fit_score(tr, te)
        for lo, hi in BANDS:
            m = (age[te] >= lo) & (age[te] < hi)
            if m.sum() < 20:
                continue
            idx, sc = te[m], s[m]; k = max(1, int(round(budget * len(idx))))
            order = np.argsort(-sc if mode == 'flag' else sc, kind='stable')
            sel = np.zeros(len(idx), bool); sel[order[:k]] = True
            auc = roc_auc_score(y[idx], sc) if 0 < y[idx].sum() < len(idx) else np.nan
            recs.append(pd.DataFrame(dict(anchor=str(a.date()), band=f'{lo}-{hi}', band_mid=(lo + hi) / 2, y=y[idx], sel=sel.astype(int),
                                          day=ts.iloc[idx].dt.strftime('%Y-%m-%d').values, auc=auc)))
    d = pd.concat(recs, ignore_index=True); rows = []
    for (band, mid), g in d.groupby(['band', 'band_mid']):
        t = g.groupby('day').agg(E=('y', 'sum'), S=('sel', lambda v: 0)).assign(SE=g.assign(se=g.y * g.sel).groupby('day').se.sum())
        days = t.index.values; Ev, SEv = t.E.to_numpy(float), t.SE.to_numpy(float)
        est = SEv.sum() / Ev.sum()
        bs = [(lambda i: SEv[i].sum() / Ev[i].sum())(rng_boot.integers(0, len(days), len(days))) for _ in range(B)]
        rows.append(dict(band=band, band_mid=mid, n=len(g), events=int(g.y.sum()), anchors=g.anchor.nunique(),
                         value=est, lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)),
                         mean_auc=float(g.groupby('anchor').auc.first().mean())))
    return pd.DataFrame(rows).sort_values('band_mid')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--secom-zip', type=Path, required=True)
    ap.add_argument('--aoi-csv', type=Path); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--boot', type=int, default=2000); a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    rngb = np.random.default_rng(SEED); out = []
    X, y, ts = P.load(a.secom_zip)
    for name in ('logistic', 'random_forest'):
        def fs(tr, te, name=name):
            cols = P.keep_cols(X.iloc[tr]); m = P.models()[name]; m.fit(X.iloc[tr][cols], y[tr])
            return m.predict_proba(X.iloc[te][cols])[:, 1]
        r = run(ts, y, fs, P.BUDGET, 'flag', rngb, a.boot); r.insert(0, 'model', name); r.insert(0, 'dataset', 'SECOM')
        r['metric'] = 'recall_at_10pct'; r['random_value'] = P.BUDGET; out.append(r); print('SECOM', name, 'done', flush=True)
    if a.aoi_csv:
        Xa, ya, tsa = A.load(a.aoi_csv); rng = np.random.default_rng(SEED)
        def fa(tr, te):
            return A.fit(Xa, ya, tr, rng).predict_proba(Xa.iloc[te])[:, 1]
        r = run(tsa, ya, fa, A.RELEASE, 'release', rngb, a.boot); r.insert(0, 'model', 'hgb'); r.insert(0, 'dataset', 'AOI')
        r['metric'] = 'slip_at_40pct_release'; r['random_value'] = A.RELEASE; out.append(r); print('AOI done', flush=True)
    df = pd.concat(out, ignore_index=True); df.to_csv(a.output / 'model_ageing.csv', index=False)
    print(df.round(4).to_string())


if __name__ == '__main__':
    main()
