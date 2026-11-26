"""Figures and the paired Study 1 delay contrast (day-block bootstrap, same resampled days for both delays)."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False})
C = {'WS': '#4C72B0', 'FX': '#DD8452', 'AC': '#55A868', 'AE': '#8172B3',
     'logistic': '#4C72B0', 'random_forest': '#C44E52'}


def paired_delay_contrast(d1, out, B=2000):
    pr = pd.read_csv(d1 / 'study1_prequential_predictions.csv')
    rng = np.random.default_rng(20261123); rows = []
    for model, g in pr[pr.delay_days >= 0].groupby('model'):
        w = g.pivot_table(index=['row', 'day', 'y'], columns='delay_days', values='flagged').reset_index()
        w = w[w.y == 1]
        days = w.day.unique(); groups = {k: v for k, v in w.groupby('day')}
        for L in (1, 3, 7, 14):
            est = w[0].mean() - w[L].mean(); bs = []
            for _ in range(B):
                s = pd.concat([groups[k] for k in rng.choice(days, len(days), replace=True)])
                bs.append(s[0].mean() - s[L].mean())
            rows.append(dict(model=model, contrast=f'recall(L=0) - recall(L={L})', estimate=est,
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             failure_days=len(days)))
    df = pd.DataFrame(rows); df.to_csv(out / 'study1_paired_delay_contrast.csv', index=False)
    return df


def fig_framework(fig_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.0)); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    def box(x, y, w, h, t, fc, fs=7.3, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04', fc=fc, ec='#444', lw=.7))
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal', linespacing=1.25)
    def arr(x1, y1, x2, y2):
        ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', lw=.8, color='#444'))
    for x, t in ((.1, 'Data'), (4.2, 'Evaluation designs'), (8.3, 'Decision tools and outputs')):
        ax.text(x, 6.7, t, fontsize=8.5, weight='bold')
    box(.1, 4.9, 3.4, 1.4, 'Study 1: SECOM\nsemiconductor line\n1,567 units, 104 failures', '#E8F0FA')
    box(.1, 3.0, 3.4, 1.4, 'Study 3: Siemens AOI\nelectronics line\n440,274 alarms', '#E8F0FA')
    box(.1, 1.1, 3.4, 1.4, 'Study 2: simulation\nheuristics vs acceptance\nsampling + CUSUM / EWMA', '#EAF5EC')
    box(4.2, 5.25, 3.5, 1.05, 'Time-ordered replay vs random\nsplit; label delays 0–14 days', '#F4F4F4')
    box(4.2, 4.0, 3.5, 1.05, 'Size-matched C / M / W designs;\nbatch and causal decisions', '#F4F4F4')
    box(4.2, 2.75, 3.5, 1.05, 'Model-ageing curves', '#F4F4F4')
    box(4.2, 1.1, 3.5, 1.3, 'Report-delay sweeps with common\nrandom numbers; cost grid;\nexposure constraints', '#F4F4F4')
    box(8.3, 4.95, 3.6, 1.35, 'Online release threshold with\ndelayed or audited feedback\n(Propositions 1 and 3)', '#FFF3E0')
    box(8.3, 1.1, 3.6, 1.35, 'Latency requirement and\ncertified maximum delay\n(Proposition 2, Corollary 2)', '#FFF3E0')
    box(8.3, 2.85, 3.6, 1.8, 'Reporting protocol (Table 8):\nvalue with exposure, decision\ntiming, ageing, matched\ncomparators, reproducibility', '#FFF3E0')
    for y in (5.6, 3.7): arr(3.5, y, 4.2, y + .1)
    arr(3.5, 1.8, 4.2, 1.75); arr(7.7, 5.0, 8.3, 5.4); arr(7.7, 3.27, 8.3, 3.6); arr(7.7, 1.75, 8.3, 1.75)
    ax.text(6.0, .45, 'Shared question: how much does evaluating with information unavailable at decision time overstate value,\n'
            'which mechanism drives it, and does it change the decision?', ha='center', fontsize=7.4, style='italic')
    fig.tight_layout(); fig.savefig(fig_dir / 'fig1_study_design.png', dpi=300); fig.savefig(fig_dir / 'fig1_study_design.pdf'); plt.close(fig)


def fig_study1(d1, fig_dir):
    s = pd.read_csv(d1 / 'study1_prequential_summary.csv'); meta = json.loads((d1 / 'study1_meta.json').read_text())
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, m in enumerate(('logistic', 'random_forest')):
        g = s[(s.design == 'prequential') & (s.model == m)].sort_values('delay_days')
        x = np.arange(len(g)) + (i - .5) * .12
        ax.errorbar(x, g.recall_at_budget, yerr=[g.recall_at_budget - g.recall_lo, g.recall_hi - g.recall_at_budget],
                    fmt='o-', color=C[m], capsize=3, lw=1.2, ms=4, label=f'{m.replace("_", " ")}: deployment replay')
        r = meta['retrospective_random_split'][m]['recall_at_budget']
        ax.axhspan(r['p2_5'], r['p97_5'], color=C[m], alpha=.08)
        ax.axhline(r['mean'], color=C[m], ls='--', lw=.9, label=f'{m.replace("_", " ")}: retrospective random split')
    ax.axhline(meta['budget'], color='k', ls=':', lw=1, label='random inspection (= budget)')
    ax.set_xticks(range(5)); ax.set_xticklabels([f'L={d} d' for d in meta['delays_days']])
    ax.set_ylabel('Share of failures caught\n(top 10% inspected)'); ax.set_xlabel('Label delay in deployment replay')
    ax.set_ylim(0, .45); ax.legend(fontsize=7, frameon=False, loc='upper right', ncol=1)
    fig.tight_layout(); fig.savefig(fig_dir / 'fig2_secom_recall_vs_delay.png', dpi=300); fig.savefig(fig_dir / 'fig2_secom_recall_vs_delay.pdf'); plt.close(fig)


def fig_study2_value(d2, fig_dir):
    s = pd.read_csv(d2 / 'study2_summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for ax, sc, ttl in zip(axes, ('nominal', 'severe_shift', 'severe_shift'), ('Nominal: net value', 'Severe shift: net value', 'Severe shift: out-of-spec periods')):
        col = 'mean_oos_periods' if 'out-of-spec' in ttl else 'mean_net'
        for p in ('WS', 'FX', 'AC', 'AE'):
            g = s[(s.scenario == sc) & (s.policy == p)].sort_values('delay')
            ax.plot(g.delay, g[col], 'o-', color=C[p], ms=3, lw=1.1, label=p)
        ax.set_title(ttl, fontsize=8.5); ax.set_xlabel('Report delay L (periods)'); ax.set_xticks(range(5))
    axes[0].set_ylabel('Mean per trajectory'); axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir / 'fig5_sim_value_vs_delay.png', dpi=300); fig.savefig(fig_dir / 'fig5_sim_value_vs_delay.pdf'); plt.close(fig)


def fig_study2_decomp(d2, fig_dir):
    d = pd.read_csv(d2 / 'study2_bias_decomposition.csv'); d = d[d.delay == 3]
    order = [('nominal', 'WS'), ('nominal', 'AC'), ('moderate_shift', 'WS'), ('moderate_shift', 'AC'),
             ('severe_shift', 'WS'), ('severe_shift', 'FX'), ('severe_shift', 'AC')]
    lab = {'nominal': 'Nominal', 'moderate_shift': 'Moderate', 'severe_shift': 'Severe'}
    rows = [d[(d.scenario == a) & (d.policy == b)].iloc[0] for a, b in order]
    fig, ax = plt.subplots(figsize=(6.6, 3.2)); x = np.arange(len(rows))
    comps = [('lost_revenue', 'Lost production (slower deployment)', '#DD8452'),
             ('extra_defect_cost', 'Extra defect cost (slower detection)', '#C44E52'),
             ('extra_pilot_cost', 'Extra pilot cost', '#8C8C8C'), ('other', 'Deploy/rollback cost', '#CCB974')]
    pos = np.zeros(len(rows)); neg = np.zeros(len(rows))
    for k, l, c in comps:
        v = np.array([r[k] for r in rows])
        ax.bar(x, np.where(v > 0, v, 0), bottom=pos, color=c, label=l, width=.6)
        ax.bar(x, np.where(v < 0, v, 0), bottom=neg, color=c, width=.6)
        pos += np.where(v > 0, v, 0); neg += np.where(v < 0, v, 0)
    ax.scatter(x, [r.total_bias for r in rows], color='k', zorder=5, s=14, label='Net overstatement')
    ax.axhline(0, color='k', lw=.6)
    ax.set_xticks(x); ax.set_xticklabels([f'{lab[a]}\n{b}' for a, b in order], fontsize=8)
    ax.set_ylabel('V(L=0) − V(L=3), normalized units'); ax.legend(fontsize=7, frameon=False, loc='upper left', ncol=2)
    ax.set_ylim(min(neg.min(), 0) - 2, pos.max() * 1.35)
    fig.tight_layout(); fig.savefig(fig_dir / 'fig6_bias_decomposition.png', dpi=300); fig.savefig(fig_dir / 'fig6_bias_decomposition.pdf'); plt.close(fig)


def fig_study3(d3, fig_dir):
    w = pd.read_csv(d3 / 'study3_weekly.csv'); meta = json.loads((d3 / 'study3_meta.json').read_text())
    fig, ax = plt.subplots(figsize=(6.6, 3.2)); ax2 = ax.twinx()
    g0 = w[w.design == '0'].sort_values('week'); gf = w[w.design == 'frozen'].sort_values('week'); x = np.arange(len(g0))
    ax2.bar(x, g0.defects / g0.n * 100, color='#DDDDDD', width=.6, zorder=0); ax2.set_ylabel('True-defect share of alarms (%)', color='#777')
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.plot(x, g0.slip_thr * 100, 'o-', color='#C44E52', ms=4, lw=1.3, label='Replay, retrained weekly (L = 0)')
    ax.plot(x, gf.slip_thr * 100, 's--', color='#4C72B0', ms=3.5, lw=1, label='Replay, frozen model')
    r = meta['retrospective']['slip_thr']
    ax.axhspan(r['p2_5'] * 100, r['p97_5'] * 100, color='#55A868', alpha=.15)
    ax.axhline(r['mean'] * 100, color='#55A868', ls='--', lw=1, label='Random-split estimate (mean, range)')
    ax.axhline(meta['target_slip'] * 100, color='k', ls=':', lw=1, label='Slip target (1%)')
    ax.set_xticks(x); ax.set_xticklabels([f'W{i+1}' for i in x]); ax.set_xlabel('Deployment week')
    ax.set_ylabel('Slip: true defects released (%)'); ax.set_ylim(bottom=0); ax2.set_ylim(bottom=0); ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.spines['right'].set_visible(True)
    fig.tight_layout(); fig.savefig(fig_dir / 'fig3_aoi_weekly_slip.png', dpi=300); fig.savefig(fig_dir / 'fig3_aoi_weekly_slip.pdf'); plt.close(fig)


def fig_ageing(res, fig_dir):
    d = pd.read_csv(res / 'ageing' / 'model_ageing.csv'); meta = json.loads((res / 'study3' / 'study3_meta.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    for i, m in enumerate(('logistic', 'random_forest')):
        g = d[(d.dataset == 'SECOM') & (d.model == m)].sort_values('band_mid'); x = g.band_mid + (i - .5) * 1.2
        ax.errorbar(x, g.value, yerr=[g.value - g.lo, g.hi - g.value], fmt='o-', color=C[m], capsize=2, ms=4, lw=1.1,
                    label=m.replace('_', ' '))
    ax.axhline(.10, color='k', ls=':', lw=1, label='random inspection'); ax.set_ylim(bottom=0)
    ax.set_title('SECOM: share of failures caught (top 10%)', fontsize=8.5); ax.set_xlabel('Model age (days)')
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]; g = d[d.dataset == 'AOI'].sort_values('band_mid')
    ax.errorbar(g.band_mid, g.value * 100, yerr=[(g.value - g.lo) * 100, (g.hi - g.value) * 100], fmt='o-', color='#C44E52', capsize=2, ms=4, lw=1.1, label='gradient boosting')
    ax.axhline(meta['retrospective']['slip_fixed']['mean'] * 100, color='#55A868', ls='--', lw=1, label='random-split estimate')
    ax.set_ylim(0, max(10, (g.hi.max() * 100) + 1))
    ax.set_title('AOI: slip when 40% of alarms are released (%)', fontsize=8.5); ax.set_xlabel('Model age (days)')
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    for a in axes: a.set_xticks([3.5, 10.5, 17.5, 24.5, 35]); a.set_xticklabels(['0–7', '7–14', '14–21', '21–28', '28–42'])
    fig.tight_layout(); fig.savefig(fig_dir / 'fig4_model_ageing.png', dpi=300); fig.savefig(fig_dir / 'fig4_model_ageing.pdf'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--study1', type=Path, required=True)
    ap.add_argument('--study2', type=Path, required=True); ap.add_argument('--figures', type=Path, required=True)
    a = ap.parse_args(); a.figures.mkdir(parents=True, exist_ok=True)
    print(paired_delay_contrast(a.study1, a.study1).round(3).to_string())
    fig_framework(a.figures); fig_study1(a.study1, a.figures)
    fig_study2_value(a.study2, a.figures); fig_study2_decomp(a.study2, a.figures)
    if (a.study1 / 'study3' / 'study3_weekly.csv').exists():
        fig_study3(a.study1 / 'study3', a.figures)
    if (a.study1 / 'ageing' / 'model_ageing.csv').exists() and (a.study1 / 'study3' / 'study3_meta.json').exists():
        fig_ageing(a.study1, a.figures)


if __name__ == '__main__':
    main()
