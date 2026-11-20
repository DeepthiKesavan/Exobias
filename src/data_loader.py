from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file as text. Is it a CSV file?")


def load_koi_csv(source: Union[str, Path, io.IOBase, object]) -> pd.DataFrame:
    """Load a KOI CSV from a path or a file-like object (e.g. Streamlit upload).

    Raises ``ValueError`` with a human-readable message on any failure.
    """
    try:
        if hasattr(source, "read"):
            raw = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
        else:
            path = Path(source)
            if not path.exists():
                raise ValueError(f"File not found: {path}")
            raw = path.read_bytes()
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Could not read the file: {exc}") from exc

    if not raw or not raw.strip():
        raise ValueError("The file is empty.")

    text = _decode(raw)
    lines = text.splitlines()
    first_data = 0
    while first_data < len(lines) and (
        lines[first_data].startswith("#") or not lines[first_data].strip()
    ):
        first_data += 1
    body = "\n".join(lines[first_data:])
    if not body.strip():
        raise ValueError("The file contains only comment lines and no table.")

    try:
        df = pd.read_csv(io.StringIO(body), low_memory=False)
    except pd.errors.ParserError as exc:
        raise ValueError(f"The file is not a valid CSV table: {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The file contains no data columns.") from exc

    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError("The CSV has a header but no data rows.")
    if df.shape[1] < 3:
        raise ValueError(
            "The CSV has fewer than 3 columns; this does not look like a KOI table."
        )
    return df


def list_local_csvs(data_dir: Path = DATA_DIR) -> list[Path]:
    """CSV files available in the project's ``data/`` folder."""
    if not data_dir.exists():
        return []
    return sorted(p for p in data_dir.glob("*.csv") if p.is_file())
