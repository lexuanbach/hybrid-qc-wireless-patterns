#!/usr/bin/env python3
"""Fast structural integrity checks for regenerated submission outputs."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
_PAPER = ROOT / "submission" / "paper"
GENERATED = (_PAPER if _PAPER.is_dir() else ROOT / "output") / "generated"


def load(name: str, minimum_rows: int) -> pd.DataFrame:
    path = RESULTS / name
    if not path.is_file():
        raise AssertionError(f"missing result: {path}")
    frame = pd.read_csv(path)
    if len(frame) < minimum_rows:
        raise AssertionError(f"{name}: expected at least {minimum_rows} rows, got {len(frame)}")
    return frame


def require_values(frame: pd.DataFrame, column: str, values: set[object], name: str) -> None:
    observed = set(frame[column].dropna().unique())
    missing = values - observed
    if missing:
        raise AssertionError(f"{name}: {column} misses {sorted(missing, key=str)}")


def require_macros(name: str, macros: set[str]) -> None:
    path = GENERATED / name
    if not path.is_file():
        raise AssertionError(f"missing generated macro file: {path}")
    text = path.read_text(encoding="utf-8")
    missing = {macro for macro in macros if f"\\newcommand{{\\{macro}}}" not in text}
    if missing:
        raise AssertionError(f"{name}: missing macros {sorted(missing)}")


def main() -> None:
    wireless = load("wireless_ntn.csv", 4_000)
    require_values(wireless, "policy", {"queue-aware", "queue-blind", "always-qpu", "always-local"}, "wireless")
    require_values(wireless, "ran_evidence", {"measured"}, "wireless")
    require_values(wireless, "radio_evidence", {"modeled"}, "wireless")

    p1 = load("pattern1_offload.csv", 15_000)
    require_values(p1, "policy", {"P1", "utility-only", "logged-IPW", "LinUCB", "Pareto", "oracle", "train-selected fixed"}, "P1")

    p2 = load("pattern2_federation.csv", 400)
    require_values(p2, "dataset", {"cancer", "digits", "wine", "ran-5g"}, "P2")
    require_values(p2, "method", {"best local", "FedDF", "FedMD-1r", "FedMD-2r", "FedMD", "centralized reference"}, "P2")
    sensitivity = load("pattern2_sensitivity.csv", 2_000)
    require_values(sensitivity, "noise_lambda", {0.0, 0.45, 0.75}, "P2 sensitivity")
    require_values(sensitivity, "public_fraction", {0.12, 0.24, 0.36}, "P2 sensitivity")

    p3 = load("pattern3_subgraph.csv", 3_000)
    require_values(p3, "scope", {"full", "true", "inferred", "wrong", "inferred-noise"}, "P3")
    require_values(p3, "noise_lambda", {0.0, 0.1, 0.25, 0.5}, "P3")
    p3_scale = load("pattern3_scope_scale.csv", 600)
    require_values(p3_scale, "n", {1_000, 5_000, 10_000}, "P3 scaling")

    p4 = load("pattern4_randomized.csv", 50_000)
    require_values(p4, "week", {"X01", "X02", "X03"}, "P4")
    require_values(p4, "controller", {"unguarded", "typed", "authenticated", "verified", "guarded", "guarded-stable"}, "P4")
    require_values(p4, "fault", {"none", "signed-bias", "replay", "corruption", "planner-bypass"}, "P4")
    load("pattern4_fault_breakdown.csv", 54)

    require_macros("pattern1.tex", {"PatOneCalSmall", "PatOneIpwLarge"})
    require_macros("wireless_ntn.tex", {"WirelessAwareSuccess", "WirelessQueueTail"})
    require_macros("pattern2.tex", {"PatTwoFedMDOne", "PatTwoFedMDTwo", "PatTwoFedMD"})
    require_macros("pattern3.tex", {"PatThreeNoiseHighPOpt", "PatThreeScopeTenKTailMs"})
    require_macros("pattern4.tex", {"IntegratedRandomUnsafeFull", "IntegratedVerifyTailUs"})
    require_macros("pattern_stats.tex", {"PatTwoDatasetDiff", "PatTwoNBRan"})
    print("result integrity: OK")


if __name__ == "__main__":
    main()
