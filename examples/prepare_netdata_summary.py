#!/usr/bin/env python3
"""Create the compact NetData 5G summary bundled with this artifact.

The upstream CSV does not declare a redistribution license.  Consequently,
the artifact keeps only aggregate statistics for 12 cells and records the
raw object's hash and deterministic selection rule.  This script is not part
of ``make reproduce`` because reproduction starts from the bundled summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = (
    "https://media.githubusercontent.com/media/tsinghua-fib-lab/NetData/"
    "main/Performance_5G_Weekday.csv"
)
SOURCE_COMMIT = "6c796243f896a4fc6b1b4236348c9af8722d397a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", type=Path)
    args = parser.parse_args()
    raw = args.raw_csv
    columns = ["Base Station ID", "Cell ID", "Timestamp",
               "PRB Usage Ratio (%)", "Traffic Volume (KByte)",
               "Number of Users"]
    data = pd.read_csv(raw, usecols=columns)
    for column in columns[3:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()

    counts = data.groupby("Base Station ID")["Cell ID"].nunique()
    candidates = counts[counts >= 3].index
    traffic = data[data["Base Station ID"].isin(candidates)].groupby(
        "Base Station ID")["Traffic Volume (KByte)"].mean().sort_values(
            ascending=False)
    stations = list(traffic.head(4).index)
    cells: list[str] = []
    for station in stations:
        ranked = data[data["Base Station ID"] == station].groupby("Cell ID")[
            "Traffic Volume (KByte)"].mean().sort_values(ascending=False)
        cells.extend(ranked.head(3).index.astype(str).tolist())

    subset = data[data["Cell ID"].astype(str).isin(cells)]
    summary = subset.groupby("Timestamp").agg(
        prb_mean_pct=("PRB Usage Ratio (%)", "mean"),
        prb_p95_pct=("PRB Usage Ratio (%)", lambda x: x.quantile(0.95)),
        traffic_total_kbyte=("Traffic Volume (KByte)", "sum"),
        users_total=("Number of Users", "sum"),
        cells=("Cell ID", "nunique"),
    ).reset_index()
    summary["minute"] = summary.Timestamp.str.slice(0, 2).astype(int) * 60 \
        + summary.Timestamp.str.slice(3, 5).astype(int)
    summary = summary.sort_values("minute")
    if len(summary) != 48 or (summary.cells != 12).any():
        raise RuntimeError("expected 48 complete half-hour windows over 12 cells")
    output = ROOT / "data" / "netdata_5g_summary.csv"
    summary.to_csv(output, index=False, float_format="%.8f")
    cells_frame = subset.groupby(["Base Station ID", "Cell ID", "Timestamp"]).agg(
        prb_pct=("PRB Usage Ratio (%)", "mean"),
        traffic_kbyte=("Traffic Volume (KByte)", "mean"),
        users=("Number of Users", "mean"),
    ).reset_index()
    cells_frame["minute"] = cells_frame.Timestamp.str.slice(0, 2).astype(int) * 60 \
        + cells_frame.Timestamp.str.slice(3, 5).astype(int)
    cells_frame = cells_frame.sort_values(["Cell ID", "minute"])
    if len(cells_frame) != 576:
        raise RuntimeError("expected 576 cell-window rows")
    cells_frame.to_csv(ROOT / "data" / "netdata_5g_cells.csv", index=False,
                       float_format="%.8f")
    provenance = {
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "raw_rows": int(len(data)),
        "selected_stations": [str(value) for value in stations],
        "selected_cells": cells,
        "aggregation": (
            "48 half-hour network aggregates plus 576 cell-window means; "
            "mean/p95 PRB and traffic/user fields"
        ),
        "license_note": (
            "The upstream repository declared no license when accessed; the raw "
            "CSV is not redistributed."
        ),
    }
    (ROOT / "data" / "netdata_5g_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
