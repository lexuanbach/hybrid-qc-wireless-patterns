#!/usr/bin/env python3
"""Cluster-paired bootstrap intervals and cross-study diagnostic macros."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t

from _common import RES, write_macros


def bootstrap_units(values: np.ndarray, draws: int = 20_000,
                    seed: int = 0) -> tuple[float, float, float]:
    """Percentile interval after the caller has reduced data to independent units."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


macros = {}

# P1: reduce all contexts within a workload seed before resampling 24 seeds.
p1 = pd.read_csv(RES / "pattern1_offload.csv")
for n, tag in ((12, "Small"), (20, "Large")):
    pivot = p1[p1.n == n].pivot_table(index=["seed", "context"], columns="policy",
                                      values="utility")
    context_diff = pivot["P1"] - pivot["train-selected fixed"]
    seed_diff = context_diff.groupby(level="seed").mean().to_numpy()
    mean, low, high = bootstrap_units(seed_diff, seed=n)
    macros[f"PatOneDiff{tag}"] = f"{mean:+.3f}"
    macros[f"PatOneDiffLo{tag}"] = f"{low:+.3f}"
    macros[f"PatOneDiffHi{tag}"] = f"{high:+.3f}"
    print(f"P1 n={n} policy-fixed {mean:+.3f} [{low:+.3f},{high:+.3f}]")

sensitivity = pd.read_csv(RES / "pattern1_sensitivity.csv")
sense = sensitivity.pivot_table(index=["scenario", "n"], columns="policy", values="utility")
nonnegative = int((sense["P1"] >= sense["train-selected fixed"] - 1e-12).sum())
macros["PatOneSensitivityNonnegative"] = f"{nonnegative}/{len(sense)}"

# P2: the split-level interval is descriptive because repeated holdouts from
# one dataset overlap.  The inferential interval below treats each dataset as
# one independent cluster; per-dataset Nadeau--Bengio intervals retain the
# repeated-holdout dependence correction.
p2 = pd.read_csv(RES / "pattern2_federation.csv")
pivot2 = p2.pivot_table(index=["dataset", "seed"], columns="method", values="accuracy")
fedmd_diff = pivot2["FedMD"] - pivot2["best local"]
mean, low, high = bootstrap_units(
    fedmd_diff.to_numpy(), seed=101)
macros.update({"PatTwoFedMDDiff": f"{100 * mean:+.1f}",
               "PatTwoFedMDDiffLo": f"{100 * low:+.1f}",
               "PatTwoFedMDDiffHi": f"{100 * high:+.1f}"})
dataset_diff = fedmd_diff.groupby(level="dataset").mean()
critical = float(t.ppf(0.975, len(dataset_diff) - 1))
dataset_radius = critical * float(dataset_diff.sem())
macros.update({
    "PatTwoDatasetDiff": f"{100 * dataset_diff.mean():+.1f}",
    "PatTwoDatasetDiffLo": f"{100 * (dataset_diff.mean() - dataset_radius):+.1f}",
    "PatTwoDatasetDiffHi": f"{100 * (dataset_diff.mean() + dataset_radius):+.1f}",
    "PatTwoIndependentClusters": len(dataset_diff),
})
test_train_ratio = 0.24 / 0.76
for dataset, differences in fedmd_diff.groupby(level="dataset"):
    values = differences.to_numpy()
    corrected_se = float(np.sqrt((1 / len(values) + test_train_ratio)
                                 * np.var(values, ddof=1)))
    radius = float(t.ppf(0.975, len(values) - 1)) * corrected_se
    tag = {"cancer": "Cancer", "digits": "Digits", "wine": "Wine",
           "ran-5g": "Ran"}[dataset]
    macros[f"PatTwoNB{tag}"] = f"{100 * values.mean():+.1f}"
    macros[f"PatTwoNB{tag}Lo"] = f"{100 * (values.mean() - radius):+.1f}"
    macros[f"PatTwoNB{tag}Hi"] = f"{100 * (values.mean() + radius):+.1f}"
mean, low, high = bootstrap_units(
    (pivot2["centralized reference"] - pivot2["FedMD"]).to_numpy(), seed=102)
macros.update({"PatTwoCentralGap": f"{100 * mean:.1f}",
               "PatTwoCentralGapLo": f"{100 * low:.1f}",
               "PatTwoCentralGapHi": f"{100 * high:.1f}"})
fedmd = p2[p2.method == "FedMD"]
feddf = p2[p2.method == "FedDF"]
best = p2[p2.method == "best local"]
macros.update({
    "PatTwoFedMDBytes": f"{int(round(fedmd.bytes.mean())):,}",
    "PatTwoFedDFBytes": f"{int(round(feddf.bytes.mean())):,}",
    "PatTwoFedMDECE": f"{fedmd.ece.mean():.3f}",
    "PatTwoBestECE": f"{best.ece.mean():.3f}",
})
influence = p2[p2.method == "unweighted ensemble"].set_index(["dataset", "seed"])
negative = int((influence.min_client_contribution < 0).sum())
macros["PatTwoNegativeContribution"] = f"{negative}/{len(influence)}"
p2s = pd.read_csv(RES / "pattern2_sensitivity.csv")
p2sp = p2s.pivot_table(index=["dataset", "seed", "noniid", "noise_lambda",
                              "public_size", "clients"], columns="method", values="accuracy")
nonnegative = int((p2sp["FedMD"] >= p2sp["best local"] - 1e-12).sum())
macros["PatTwoSensitivityNonnegative"] = f"{nonnegative}/{len(p2sp)}"
fedmd_sensitivity = p2s[(p2s.method == "FedMD") & (p2s.clients == 3)]
for value, tag in ((0.0, "Zero"), (0.45, "Mid"), (0.75, "High")):
    part = fedmd_sensitivity[np.isclose(fedmd_sensitivity.noise_lambda, value)]
    macros[f"PatTwoNoise{tag}Accuracy"] = f"{100 * part.accuracy.mean():.1f}\\%"
for value, tag in ((0.12, "Small"), (0.24, "Medium"), (0.36, "Large")):
    part = fedmd_sensitivity[np.isclose(fedmd_sensitivity.public_fraction, value)]
    macros[f"PatTwoPublic{tag}Accuracy"] = f"{100 * part.accuracy.mean():.1f}\\%"
print(f"P2 FedMD-best local {macros['PatTwoFedMDDiff']} pp "
      f"[{macros['PatTwoFedMDDiffLo']},{macros['PatTwoFedMDDiffHi']}]")
print(f"P2 dataset-cluster {macros['PatTwoDatasetDiff']} pp "
      f"[{macros['PatTwoDatasetDiffLo']},{macros['PatTwoDatasetDiffHi']}]")

# P3: compare scopes under the identical condition/seed and report seed-cluster
# uncertainty for the scope penalty. Reference QAOA rows are p=1, 256 shots.
p3 = pd.read_csv(RES / "pattern3_subgraph.csv")
reference = p3[(p3.patch == 6) & (p3.solver == "QAOA") &
               (p3.depth == 1) & (p3.shots == 256) &
               p3.scope.isin(["true", "inferred", "wrong", "full"])]
scope_mean = reference.groupby("scope").gap.mean()
macros["PatThreeTrueGap"] = f"{scope_mean['true']:.3f}"
macros["PatThreeWrongGap"] = f"{scope_mean['wrong']:.3f}"
pivot3 = reference.pivot_table(index=["seed", "coupling", "severity"],
                               columns="scope", values="gap")
seed_penalty = (pivot3["wrong"] - pivot3["true"]).groupby(level="seed").mean().to_numpy()
mean, low, high = bootstrap_units(seed_penalty, seed=103)
macros.update({"PatThreeScopePenalty": f"{mean:.3f}",
               "PatThreeScopePenaltyLo": f"{low:.3f}",
               "PatThreeScopePenaltyHi": f"{high:.3f}"})

# Integrated deterministic case: retain week and eight-window fault cycle.
p4 = pd.read_csv(RES / "pattern4_integrated.csv")
p4p = p4.pivot_table(index=["week", "window"], columns="controller",
                     values="unsafe_commit")
delta = (p4p["unguarded"] - p4p["guarded"]).to_numpy()
blocks = delta.reshape(-1, len(__import__("pattern4_integrated_workflow").FAULTS)).mean(axis=1)
mean, low, high = bootstrap_units(blocks, seed=104)
macros.update({"IntegratedUnsafeReduction": f"{100 * mean:.1f}",
               "IntegratedUnsafeReductionLo": f"{100 * low:.1f}",
               "IntegratedUnsafeReductionHi": f"{100 * high:.1f}"})

# Random schedules: each value is a complete replay of one measured week.
# Replications over the same week are schedule-robustness checks, not new
# independent traffic traces, so report the observed schedule-week range.
p4r = pd.read_csv(RES / "pattern4_randomized.csv")
p4ru = p4r.groupby(["replay_id", "week", "controller"]).unsafe_commit.mean().unstack()
random_delta = 100 * (p4ru["unguarded"] - p4ru["guarded-stable"])
macros.update({
    "IntegratedRandomReduction": f"{random_delta.mean():.1f}",
    "IntegratedRandomReductionMin": f"{random_delta.min():.1f}",
    "IntegratedRandomReductionMax": f"{random_delta.max():.1f}",
    "IntegratedRandomScheduleWeeks": len(random_delta),
})

write_macros("pattern_stats", macros)
