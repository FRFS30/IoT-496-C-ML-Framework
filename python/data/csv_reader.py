import math
from ..utils.math import isnan


def read_csv(path, chunk_size=50_000):
    """
    Read CSV returning dict[col_name -> list].
    Handles Inf / -Inf / empty -> NaN.
    Strips whitespace from headers (required for CIC-IDS-2017).
    Infers numeric vs string types.
    Memory-efficient: reads in chunks but accumulates full columns.
    """
    result = {}
    headers = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        chunk = []
        for lineno, line in enumerate(f):
            line = line.rstrip("\n\r")
            if lineno == 0:
                headers = [h.strip() for h in line.split(",")]
                for h in headers:
                    result[h] = []
                continue

            if line == "":
                continue

            chunk.append(line)
            if len(chunk) >= chunk_size:
                _parse_chunk(chunk, headers, result)
                chunk = []

        if chunk:
            _parse_chunk(chunk, headers, result)

    # Infer numeric columns and cast
    _infer_and_cast(result)
    return result


def _parse_chunk(lines, headers, result):
    ncols = len(headers)
    for line in lines:
        parts = line.split(",")
        # pad / truncate to match header count
        while len(parts) < ncols:
            parts.append("")
        for i, h in enumerate(headers):
            result[h].append(parts[i].strip())


def _parse_float(s):
    if s in ("", "nan", "NaN", "NAN"):
        return float("nan")
    s_upper = s.upper()
    if s_upper in ("INF", "INFINITY", "+INF", "+INFINITY"):
        return float("nan")
    if s_upper in ("-INF", "-INFINITY"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _looks_numeric(values, sample=200):
    """Return True if most sampled values parse as float."""
    tested = values[:sample]
    if not tested:
        return False
    numeric_count = 0
    for v in tested:
        try:
            float(v)
            numeric_count += 1
        except (ValueError, TypeError):
            pass
    return numeric_count / len(tested) >= 0.8


def _infer_and_cast(result):
    for key, values in result.items():
        if _looks_numeric(values):
            result[key] = [_parse_float(v) for v in values]
