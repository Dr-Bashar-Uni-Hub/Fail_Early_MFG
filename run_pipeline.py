"""Reproduce every result, table number and figure in the manuscript.

    python scripts/get_secom.py                 # downloads the UCI SECOM archive to data/raw/secom.zip
    python run_pipeline.py                      # ~10 min on a laptop
    python run_pipeline.py --sensitivity        # adds Section 5.5 sensitivity analyses (~20 min more)
    python run_pipeline.py --quick              # smoke test (fewer seeds / bootstrap draws)
"""
import argparse, hashlib, json, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args):
    print('>>', ' '.join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT / 'src')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--secom-zip', type=Path, default=ROOT / 'data/raw/secom.zip')
    ap.add_argument('--results', type=Path, default=ROOT / 'results')
    ap.add_argument('--figures', type=Path, default=ROOT / 'figures')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--aoi-csv', type=Path, help='Siemens AOI dataset.csv (Mendeley Data doi:10.17632/99jzmh9658.1) for Study 3')
    ap.add_argument('--sensitivity', action='store_true', help='also run the sensitivity analyses (~20 min)')
    a = ap.parse_args()
    if not a.secom_zip.exists():
        ap.error(f'{a.secom_zip} not found; run python scripts/get_secom.py or pass --secom-zip')
    seeds, boot, reps = (100, 200, 2000) if a.quick else (1000, 2000, 20000)
    a.results.mkdir(parents=True, exist_ok=True)
    run('secom_prequential.py', '--secom-zip', a.secom_zip.resolve(), '--output', a.results.resolve(), '--boot', boot)
    run('delay_sim.py', '--output', a.results.resolve(), '--seeds', seeds)
    run('chart_arl.py', '--output', a.results.resolve(), '--reps', reps)
    if a.aoi_csv:
        run('aoi_replay.py', '--data', a.aoi_csv.resolve(), '--output', (a.results / 'study3').resolve(),
            '--boot', 200 if a.quick else 2000, '--repeats', 2 if a.quick else 10)
        run('aoi_methods.py', '--data', a.aoi_csv.resolve(), '--output', (a.results / 'study3').resolve(),
            '--boot', 200 if a.quick else 2000, '--repeats', 1 if a.quick else 3)
        run('aoi_online.py', '--data', a.aoi_csv.resolve(), '--output', (a.results / 'study3').resolve())
        run('aoi_selective.py', '--data', a.aoi_csv.resolve(), '--output', (a.results / 'study3').resolve(), '--D', 7)
        run('model_ageing.py', '--secom-zip', a.secom_zip.resolve(), '--aoi-csv', a.aoi_csv.resolve(),
            '--output', (a.results / 'ageing').resolve(), '--boot', 200 if a.quick else 2000)
    run('risk_constrained.py', '--results', a.results.resolve())
    run('exposure_latency.py', '--results', a.results.resolve())
    run('latency_certification.py', '--output', a.results.resolve(), '--n-ref', 2000 if a.quick else 20000, '--reps', 100 if a.quick else 1000)
    run('leakage_decomposition.py', '--secom-zip', a.secom_zip.resolve(), *(['--aoi-csv', a.aoi_csv.resolve()] if a.aoi_csv else []),
        '--output', (a.results / 'decomposition').resolve(), '--boot', 200 if a.quick else 2000)
    if a.aoi_csv and not a.quick:
        for k in (1, 2):
            run('leakage_decomposition.py', '--secom-zip', a.secom_zip.resolve(), '--aoi-csv', a.aoi_csv.resolve(),
                '--output', (a.results / 'decomposition').resolve(), '--boot', 1000, '--seed', k)
    if a.sensitivity:
        run('sensitivity.py', '--part', 'all', '--secom-zip', a.secom_zip.resolve(), '--output', (a.results / 'sensitivity').resolve(),
            '--seeds', 100 if a.quick else 500, '--boot', 100 if a.quick else 500, '--repeats', 5 if a.quick else 20)
    run('make_figures.py', '--study1', a.results.resolve(), '--study2', a.results.resolve(), '--figures', a.figures.resolve())
    run('make_tables.py', '--results', a.results.resolve())
    with zipfile.ZipFile(a.secom_zip) as z:
        inputs = {n: hashlib.sha256(z.read(n)).hexdigest() for n in ('secom.data', 'secom_labels.data')}
    (a.results / 'run_record.json').write_text(json.dumps(dict(
        quick=a.quick, seeds=seeds, bootstrap=boot, arl_reps=reps, python=sys.version.split()[0],
        input_sha256=inputs), indent=2))
    print('DONE. Results in', a.results, '| figures in', a.figures)


if __name__ == '__main__':
    main()
