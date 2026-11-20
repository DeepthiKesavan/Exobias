"""Download the Kepler DR25 KOI table from the NASA Exoplanet Archive into data/koi_dr25.csv.

Usage (from the folder containing ExoBias.py, with your internet connection on):
    python download_dataset.py

The file is validated after download (columns, row count, class counts). Nothing is generated or
altered: it is saved exactly as NASA's archive returns it.
"""
import argparse
import csv
import io
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

DEFAULT_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+q1_q17_dr25_koi&format=csv"
OUT_PATH = Path(__file__).resolve().parent / "data" / "koi_dr25.csv"
RECOMMENDED = ["kepid", "kepoi_name", "koi_disposition", "koi_score", "koi_period", "koi_prad",
               "koi_teq", "koi_insol", "koi_steff", "koi_srad", "koi_smass"]


def download(url: str, timeout: int = 300) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ExoBias-dataset-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def summarise(raw: bytes) -> dict:
    """Parse the CSV (skipping '#' comment lines) and report what it contains."""
    text = raw.decode("utf-8-sig", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise ValueError("The download is empty.")
    if lines[0].lstrip().startswith("<"):
        raise ValueError("The server returned a web page or an error message instead of CSV data:\n" + lines[0][:300])
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    columns = reader.fieldnames or []
    if "koi_disposition" not in columns:
        raise ValueError("The file has no 'koi_disposition' column, so it is not a KOI table.")
    rows, classes = 0, Counter()
    for row in reader:
        rows += 1
        classes[(row.get("koi_disposition") or "").strip() or "(missing)"] += 1
    return {"rows": rows, "columns": columns, "classes": dict(classes),
            "missing_recommended": [c for c in RECOMMENDED if c not in columns]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL, help="override the download URL")
    parser.add_argument("--out", default=str(OUT_PATH), help="output CSV path")
    args = parser.parse_args()

    print("Downloading from the NASA Exoplanet Archive (this can take a minute)...")
    try:
        raw = download(args.url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"\nDownload failed: {exc}\n\nOpen this link in your browser instead and save the file as data/koi_dr25.csv:\n{args.url}")
        return 1
    try:
        info = summarise(raw)
    except ValueError as exc:
        print(f"\nThe downloaded data is not usable: {exc}")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    print(f"\nSaved {len(raw) / 1e6:.1f} MB to {out}")
    print(f"Rows: {info['rows']:,}   Columns: {len(info['columns'])}")
    print("Class counts:", info["classes"])
    if info["missing_recommended"]:
        print("Recommended columns NOT in this file:", ", ".join(info["missing_recommended"]))
    else:
        print("All recommended columns are present.")
    print("\nNext: run `streamlit run ExoBias.py` and choose this file on the Data Explorer page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
