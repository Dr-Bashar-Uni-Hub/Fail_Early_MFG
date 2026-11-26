"""Operating characteristics (average run length, in batches of 20) of the Study 2 monitoring charts."""
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
import delay_sim as S

def run_length(p, chart, rng, cap=5000):
    s, z = 0., S.P0
    for t in range(1, cap + 1):
        k = rng.binomial(S.B, p)
        if chart == 'CUSUM':
            s = max(0., s + k * math.log(S.P1 / S.P0) + (S.B - k) * math.log((1 - S.P1) / (1 - S.P0)))
            if s > S.H_CUSUM: return t
        else:
            z = S.LAM * k / S.B + (1 - S.LAM) * z
            if z > S.EWMA_UCL: return t
    return cap

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); ap.add_argument('--reps', type=int, default=20000)
    a = ap.parse_args(); rng = np.random.default_rng(99); rows = []
    for chart in ('CUSUM', 'EWMA'):
        for p in (.035, .06, .10, .12, .20, .24):
            rl = np.array([run_length(p, chart, rng) for _ in range(a.reps)])
            rows.append(dict(chart=chart, p=p, arl=rl.mean(), se=rl.std(ddof=1) / math.sqrt(len(rl)), median_rl=np.median(rl), p_signal_within_12=(rl <= 12).mean()))
    df = pd.DataFrame(rows); df.to_csv(a.output / 'study2_chart_arl.csv', index=False); print(df.round(3).to_string())

if __name__ == '__main__':
    main()
