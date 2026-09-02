#!/usr/bin/env python3
"""Ship-state gate: no manuscript files, required inputs/results present,
simulator importable, stored CSVs structurally sound."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
skip = {"output", ".venv", "__pycache__"}
bad = [p for p in root.rglob("*")
       if p.suffix in (".tex", ".pdf", ".cls", ".bst")
       and not any(part in skip for part in p.relative_to(root).parts)]
assert not bad, f"manuscript leak: {bad[:3]}"

required = [
    "examples/sim_core.py", "examples/_common.py",
    "examples/pattern1_offload_decision.py", "examples/wireless_ntn_study.py",
    "examples/pattern2_heterogeneous_federation.py",
    "examples/pattern3_subgraph_encoding.py",
    "examples/pattern4_integrated_workflow.py", "examples/pattern_stats.py",
    "examples/check_results.py",
    "data/abilene_three_weeks_summary.csv", "data/netdata_5g_summary.csv",
    "data/abilene_three_weeks_provenance.json", "data/netdata_5g_provenance.json",
    "schemas/resource-record-v1.schema.json",
    "results/pattern1_offload.csv", "results/pattern2_federation.csv",
    "results/pattern3_subgraph.csv", "results/pattern4_randomized.csv",
    "results/wireless_ntn.csv",
    "requirements-lock.txt", "README.md", "REPRODUCE.md",
]
missing = [r for r in required if not (root / r).exists()]
assert not missing, f"missing: {missing}"

sys.path.insert(0, str(root / "examples"))
import sim_core  # noqa: F401  standalone import, no sibling project

import pandas as pd
for name, rows in [("pattern1_offload.csv", 15_000),
                   ("pattern2_federation.csv", 400),
                   ("pattern3_subgraph.csv", 3_000),
                   ("pattern4_randomized.csv", 50_000),
                   ("wireless_ntn.csv", 4_000)]:
    n = len(pd.read_csv(root / "results" / name))
    assert n >= rows, f"{name}: {n} rows < {rows}"
print("artifact check: OK")
