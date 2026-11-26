"""Download the UCI SECOM dataset (McCann & Johnston, 2008; CC BY 4.0; doi:10.24432/C54305)
and repackage the two data files as data/raw/secom.zip. Checks shape and failure count."""
import io, sys, urllib.request, zipfile
from pathlib import Path

URL = 'https://archive.ics.uci.edu/static/public/179/secom.zip'
OUT = Path(__file__).resolve().parents[1] / 'data/raw/secom.zip'


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print('downloading', URL)
    raw = urllib.request.urlopen(URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = {Path(n).name: n for n in z.namelist()}
        files = {k: z.read(names[k]) for k in ('secom.data', 'secom_labels.data')}
    lab = files['secom_labels.data'].decode().strip().splitlines()
    assert len(lab) == 1567 and sum(l.split()[0] == '1' for l in lab) == 104, 'unexpected SECOM contents'
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
        for k, v in files.items():
            z.writestr(k, v)
    print('wrote', OUT)


if __name__ == '__main__':
    sys.exit(main())
