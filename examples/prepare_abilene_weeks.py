#!/usr/bin/env python3
"""Derive compact summaries from one or more Abilene X*.gz weeks."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WEEK_START = {"X01": "2004-03-01", "X02": "2004-03-08",
              "X03": "2004-04-02"}
SOURCE_BASE = "https://www.cs.utexas.edu/~yzhang/research/AbileneTM"


def read_real_od(path: Path) -> np.ndarray:
    rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            values = np.fromstring(line, sep=" ")
            if len(values) != 720:
                raise RuntimeError(f"{path}: expected 720 fields, got {len(values)}")
            rows.append(values[0::5])
    matrix = np.asarray(rows)
    if matrix.shape != (2016, 144):
        raise RuntimeError(f"{path}: expected (2016,144), got {matrix.shape}")
    return matrix


def summarize(path: Path) -> pd.DataFrame:
    week = path.name.split(".")[0]
    if week not in WEEK_START:
        raise KeyError(f"no start date declared for {week}")
    matrix = read_real_od(path)
    total = matrix.sum(axis=1)
    median = float(np.median(total))
    selected = np.linspace(0, len(matrix) - 1, 240).round().astype(int)
    start = datetime.fromisoformat(WEEK_START[week])
    rows = []
    for source in selected:
        current = matrix[source]
        next_source = min(source + 1, len(matrix) - 1)
        current_total = float(total[source] / median)
        next_total = float(total[next_source] / median)
        mean = float(np.mean(current) / median)
        sd = float(np.std(current) / median)
        rows.append({
            "week": week,
            "timestamp": (start + timedelta(minutes=5 * int(source))).isoformat(),
            "source_window": int(source), "total_load": current_total,
            "mean_od": mean, "p95_od": float(np.quantile(current, .95) / median),
            "max_od": float(np.max(current) / median),
            "cv_od": sd / (mean + 1e-12), "next_total_load": next_total,
            "relative_change": (next_total - current_total) / (current_total + 1e-12),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weeks", nargs="+", type=Path)
    args = parser.parse_args()
    frames, provenance = [], []
    for path in args.weeks:
        frames.append(summarize(path))
        provenance.append({
            "week": path.name.split(".")[0],
            "source_url": f"{SOURCE_BASE}/{path.name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    output = ROOT / "data" / "abilene_three_weeks_summary.csv"
    pd.concat(frames, ignore_index=True).to_csv(output, index=False,
                                                float_format="%.8f")
    (ROOT / "data" / "abilene_three_weeks_provenance.json").write_text(
        json.dumps({"weeks": provenance,
                    "aggregation": "240 evenly spaced windows per 2016-window week"},
                   indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
