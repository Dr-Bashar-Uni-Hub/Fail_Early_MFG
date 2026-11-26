"""Decomposing the gap between random-split estimates and deployment, with strictly causal decisions.

For every weekly deployment window w (same windows as the replays, immediate labels):
  C  chronological : model fitted on the older 80% of all units before w; threshold set on the most recent 20%.
  M  size-matched  : model fitted on a random sample of the SAME size drawn from all units outside w (earlier and
                     later production); threshold set on a random 20% of that sample.
C and M are evaluated on the same units of w, so their difference isolates the effect of using later production
(temporal leakage and drift) at equal training size and on the same evaluation population. The difference between
the usual random-split estimate R and M reflects the different evaluation population and training-set size.

Decisions (all causal: thresholds come only from the calibration data, never from scores of the window itself):
  SECOM  flag a unit if its score >= the 90th percentile of calibration scores (target inspection rate 10%);
         outcome recall and realized inspection rate. Batch version (top 10% of the window) reported for reference.
  AOI    release if score < the 40th percentile of calibration scores (target release 40%); outcome slip and realized
         release rate. Also the 1%-slip rule (already causal). Batch version (lowest 40% of window) for reference.
Uncertainty: calendar-day block bootstrap, paired between C and M.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import secom_prequential as P
import aoi_replay as A

SEED = 20261123


def windows(ts, burn):
    start = int(burn * len(ts)); t0 = ts.iloc[start].normalize()
    edges = pd.date_range(t0, ts.iloc[-1] + pd.Timedelta(days=7), freq='7D')
    for a, b in zip(edges[:-1], edges[1:]):
        win = np.where((ts >= a) & (ts < b) & (np.arange(len(ts)) >= start))[0]
        if len(win):
            yield a, win


def split_train(design, win, te, rng, y):
    """C: past only. M: size-matched random sample from all units outside the window (earlier and later weeks).
    W: size-matched random sample from all units except the evaluated half-window (may include same-week units)."""
    first = win[0]; past = np.arange(first)
    if design == 'C':
        n_cal = int(.2 * len(past)); return past[:-n_cal], past[-n_cal:]
    excl = win if design == 'M' else te
    pool = np.setdiff1d(np.arange(len(y)), excl); samp = rng.choice(pool, len(past), replace=False)
    rng.shuffle(samp); n_cal = int(.2 * len(samp)); return np.sort(samp[n_cal:]), np.sort(samp[:n_cal])


def run_secom(zip_path, rng):
    X, y, ts = P.load(zip_path); rows = []
    for model in ('logistic', 'random_forest'):
        for a, win0 in windows(ts, P.BURN_IN):
            win = np.sort(rng.choice(win0, len(win0) // 2, replace=False))      # evaluated half of the week
            for design in ('C', 'M', 'W'):
                tr, cal = split_train(design, win0, win, rng, y)
                cols = P.keep_cols(X.iloc[tr]); m = P.models()[model]; m.fit(X.iloc[tr][cols], y[tr])
                sc_cal = m.predict_proba(X.iloc[cal][cols])[:, 1]; sc = m.predict_proba(X.iloc[win][cols])[:, 1]
                thr = np.quantile(sc_cal, 1 - P.BUDGET)
                k = max(1, int(round(P.BUDGET * len(win)))); top = np.zeros(len(win), bool); top[np.argsort(-sc, kind='stable')[:k]] = True
                rows.append(pd.DataFrame(dict(dataset='SECOM', model=model, design=design, week=str(a.date()),
                                              day=ts.iloc[win].dt.strftime('%Y-%m-%d').values, y=y[win],
                                              act_causal=(sc >= thr).astype(int), act_batch=top.astype(int))))
        print('SECOM', model, 'done', flush=True)
    return pd.concat(rows, ignore_index=True)


def run_aoi(csv, rng):
    X, y, ts = A.load(csv); rows = []
    for a, win0 in windows(ts, A.BURN_IN):
        win = np.sort(rng.choice(win0, len(win0) // 2, replace=False))
        for design in ('C', 'M', 'W'):
            tr, cal = split_train(design, win0, win, rng, y)
            m = A.fit(X, y, tr, rng); sc_cal = m.predict_proba(X.iloc[cal])[:, 1]; sc = m.predict_proba(X.iloc[win])[:, 1]
            thr40 = np.quantile(sc_cal, A.RELEASE); thr1 = A.threshold_for_slip(sc_cal, y[cal])
            k = int(round(A.RELEASE * len(win))); low = np.zeros(len(win), bool); low[np.argsort(sc, kind='stable')[:k]] = True
            rows.append(pd.DataFrame(dict(dataset='AOI', model='hgb', design=design, week=str(a.date()),
                                          day=ts.iloc[win].dt.strftime('%Y-%m-%d').values, y=y[win],
                                          act_causal=(sc < thr40).astype(int), act_batch=low.astype(int),
                                          act_slip1=(sc < thr1).astype(int))))
        print('AOI week', a.date(), 'done', flush=True)
    return pd.concat(rows, ignore_index=True)


def summarise(d, rng, B):
    """Per dataset/model/action and design: share of positives acted on (recall, or slip for releases), action rate and
    lift = share / rate (1 = random). Paired day-block bootstrap; differences W-M (same-period proximity) and M-C
    (later production in training) on the same evaluated units."""
    out = []; acts = [c for c in d.columns if c.startswith('act_')]; D = ('C', 'M', 'W')
    for (ds, model), g in d.groupby(['dataset', 'model']):
        for act in acts:
            if g[act].isna().all():
                continue
            t = {}
            for des in D:
                h = g[g.design == des]
                t[des] = h.assign(pa=h.y * h[act]).groupby('day').agg(P=('y', 'sum'), PA=('pa', 'sum'), N=('y', 'size'), NA=(act, 'sum'))
            days = t['C'].index.values; arr = {k: t[k].loc[days].to_numpy(float) for k in D}
            def est(a):
                sh, rt = a[:, 1].sum() / a[:, 0].sum(), a[:, 3].sum() / a[:, 2].sum(); return np.array([sh, rt, sh / rt if rt > 0 else np.nan])
            pt = {k: est(arr[k]) for k in D}; bs = []
            for _ in range(B):
                i = rng.integers(0, len(days), len(days)); e = {k: est(arr[k][i]) for k in D}
                bs.append(np.concatenate([e['C'], e['M'], e['W'], e['W'] - e['M'], e['M'] - e['C']]))
            lo, hi = np.nanpercentile(np.array(bs), [2.5, 97.5], axis=0); row = dict(dataset=ds, model=model, action=act, positives=int(arr['C'][:, 0].sum()))
            names = []
            for k in D:
                names += [f'{k}_share', f'{k}_rate', f'{k}_lift']
            for k in ('WminusM', 'MminusC'):
                names += [f'{k}_share', f'{k}_rate', f'{k}_lift']
            vals = np.concatenate([pt['C'], pt['M'], pt['W'], pt['W'] - pt['M'], pt['M'] - pt['C']])
            for n_, v, l, h in zip(names, vals, lo, hi):
                row[n_] = v; row[n_ + '_lo'] = l; row[n_ + '_hi'] = h
            out.append(row)
    return pd.DataFrame(out)


def resummarise(out, B):
    d = pd.read_csv(out / 'leakage_decisions.csv.gz')
    s = summarise(d, np.random.default_rng(SEED), B); s.to_csv(out / 'leakage_decomposition.csv', index=False); return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--secom-zip', type=Path, required=True)
    ap.add_argument('--aoi-csv', type=Path); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + a.seed); tag = '' if a.seed == 0 else f'_seed{a.seed}'
    d = [run_secom(a.secom_zip, rng)]
    if a.aoi_csv:
        d.append(run_aoi(a.aoi_csv, rng))
    d = pd.concat(d, ignore_index=True); d.to_csv(a.output / f'leakage_decisions{tag}.csv.gz', index=False, compression='gzip')
    s = summarise(d, np.random.default_rng(SEED), a.boot); s.to_csv(a.output / f'leakage_decomposition{tag}.csv', index=False)
    print(s.round(4).to_string())


if __name__ == '__main__':
    main()
