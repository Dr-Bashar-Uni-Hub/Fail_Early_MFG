"""Unit and consistency tests. Run: python -m pytest tests  (or: python tests/test_core.py)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import delay_sim as S            # noqa: E402
import secom_prequential as P    # noqa: E402
import aoi_replay as A            # noqa: E402
import aoi_methods as AM          # noqa: E402


def test_delay_changes_release_time():
    """L=0 and L=1 must differ (a report from period t is first usable at t+1+L)."""
    u = np.random.default_rng([20261123, 0]).random((S.H, S.B))
    path = np.full(S.H, .035)
    r0, r1 = S.simulate(u, path, 'FX', 0), S.simulate(u, path, 'FX', 1)
    assert r1['deploy_t'] == r0['deploy_t'] + 1


def test_no_information_before_release():
    """With a delay longer than the horizon nothing is ever observed, so no policy can deploy."""
    u = np.random.default_rng([20261123, 1]).random((S.H, S.B))
    for pol in S.POL:
        assert S.simulate(u, np.full(S.H, .035), pol, S.H)['deployed'] == 0


def test_acceptance_plan_samples_full_n_with_small_batches():
    """With 10-item batches the plan must pilot four times (40 items) before deciding."""
    old = S.B; S.B = 10
    try:
        u = np.random.default_rng([20261123, 2]).random((S.H, S.B))
        r = S.simulate(u, np.full(S.H, .035), 'AC', 0)
        assert r['pilots'] == 4 and r['deployed'] == 1
    finally:
        S.B = old


def test_cost_parameters_do_not_affect_decisions():
    df = pd.DataFrame([dict(prod_periods=10, defects=7, pilots=2, deployed=1, rolled=1)])
    a, b = S.net_value(df, 35, 6).iloc[0], S.net_value(df, 70, 18).iloc[0]
    assert np.isclose(a - b, (70 - 35) * 7 / S.B + (18 - 6))


def test_acceptance_plan_oc():
    from scipy.stats import binom
    oc = S.acceptance_oc()
    assert np.isclose(oc['0.035'], binom.cdf(3, 40, .035)) and oc['0.240'] < .01


def test_label_availability_has_no_leakage():
    ts = pd.Series(pd.to_datetime(['2008-07-01', '2008-07-05', '2008-07-09', '2008-07-10', '2008-07-12']))
    start = pd.Timestamp('2008-07-10')
    assert list(P.available_indices(ts, 0, start, 3)) == [0, 1, 2]
    assert list(P.available_indices(ts, 3, start, 3)) == [0, 1]
    assert list(P.available_indices(ts, 14, start, 3)) == []


def test_slip_threshold_releases_at_most_target():
    rng = np.random.default_rng(3); y = (rng.random(20000) < .02).astype(int); score = rng.random(20000) + y * .5
    thr = A.threshold_for_slip(score, y, .01)
    assert (score[y == 1] < thr).mean() <= .01


def test_upper_bound_threshold_is_more_conservative():
    rng = np.random.default_rng(4); y = (rng.random(50000) < .02).astype(int); score = rng.random(50000) + y * .5
    assert AM.thr_ucb(score, y, .01) <= AM.thr_point(score, y, .01)


def test_decomposition_training_sets():
    import leakage_decomposition as LD
    rng = np.random.default_rng(5); y = np.zeros(1000, int); win0 = np.arange(600, 700); te = win0[::2]
    tr, cal = LD.split_train('C', win0, te, rng, y); assert max(tr.max(), cal.max()) < 600
    tr, cal = LD.split_train('M', win0, te, rng, y); u = np.concatenate([tr, cal]); assert not np.isin(u, win0).any() and len(u) == 600
    tr, cal = LD.split_train('W', win0, te, rng, y); u = np.concatenate([tr, cal]); assert not np.isin(u, te).any() and len(u) == 600


def test_online_slip_bound_adversarial():
    import aoi_online as O
    rng = np.random.default_rng(1)
    for trial in range(30):
        n = 5000; ts = pd.Series(pd.to_datetime('2020-01-01') + pd.to_timedelta(np.sort(rng.uniform(0, 50, n)), unit='D'))
        y = (rng.random(n) < .03).astype(int); drift = np.cumsum(rng.normal(0, .05, n))
        s = np.clip(rng.normal(0, 2, n) + drift + 3 * y, -11.9, 11.9); g = rng.choice([.01, .05, .2]); D = int(rng.choice([0, 2, 10])); th0 = rng.uniform(-11, 11)
        rel, th, e, M = O.online(s, y, ts, D, g, th0)
        assert abs(e.mean() - O.ALPHA) <= (O.B_CLIP + abs(th0) + g * (M + 1)) / (g * len(e)) + 1e-12


def test_exposure_identity_under_assumptions():
    """Fixed observation sequence: rollback at detection + L, so exposure = min(H_post, Z + L)."""
    Hp = S.H - S.SHIFT_T
    for Z in range(0, Hp + 1):
        for L in range(5):
            assert min(Hp, Z + L) - min(Hp, Z) <= L


def test_audited_selective_bound():
    import math, aoi_selective as SL, aoi_online as O
    rng = np.random.default_rng(3); viol = 0; tot = 0
    for trial in range(20):
        n = 4000; t = np.sort(rng.uniform(0, 40, n)); y = (rng.random(n) < .03).astype(int)
        s = np.clip(rng.normal(0, 2.5, n) + 2.5 * y, -11.9, 11.9); q = [.05, .2, .5][trial % 3]
        for sd in range(5):
            r, au, M = SL.online_feedback(s, y, t, .1, 0.0, D=None, q=q, rng=np.random.default_rng(sd + 10 * trial))
            N = y.sum(); lg = math.log(2 / .05)
            b = (O.B_CLIP + .1 * max(.01, .99 * M / q)) / (.1 * N) + .99 * (2 * lg / (3 * q) + math.sqrt(2 * N * (1 - q) / q * lg)) / N
            viol += abs(r[y == 1].mean() - .01) > b; tot += 1
    assert viol <= .05 * tot


def test_dkw_bound_dominates_mean():
    import latency_certification as LC
    U = np.array([-1, 0, 1, 3, 12, 2, 1, 0] * 20, float); b = LC.bounds(U, 12, .05)
    for L in b:
        assert b[L]['dkw'] >= b[L]['plugin'] and b[L]['hoeffding'] >= b[L]['plugin']


def test_secom_counts_add_up():
    R = ROOT / 'results'
    if (R / 'manuscript_numbers.json').exists():
        N = json.loads((R / 'manuscript_numbers.json').read_text())
        assert N['s1']['n_burn'] + N['s1']['n_deploy'] == 1567


def test_results_and_manuscript_numbers_consistent():
    R = ROOT / 'results'
    if not (R / 'manuscript_numbers.json').exists():
        return
    N = json.loads((R / 'manuscript_numbers.json').read_text())
    s1 = pd.read_csv(R / 'study1_prequential_summary.csv')
    row = s1[(s1.design == 'prequential') & (s1.delay_days == 0) & (s1.model == 'random_forest')].iloc[0]
    ms = [x for x in N['s1']['rows'] if x['design'] == 'prequential' and x['delay_days'] == 0 and x['model'] == 'random_forest'][0]
    assert abs(ms['recall_at_budget'] - row.recall_at_budget) < 1e-3
    raw = pd.read_csv(R / 'study2_raw.csv.gz')
    assert len(raw) == len(S.SCEN) * len(S.DELAYS) * len(S.POL) * N['s2']['meta']['seeds']


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print('PASS', name)
