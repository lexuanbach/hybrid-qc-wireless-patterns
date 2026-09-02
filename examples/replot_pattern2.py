#!/usr/bin/env python3
"""Regenerate fig_pattern2.pdf from stored results (layout-only changes)."""
import pandas as pd
from _common import RES, FIG, plt

results = pd.read_csv(RES / "pattern2_federation.csv")
METHODS = ["best local", "unweighted ensemble", "weighted ensemble",
           "hard transfer", "FedDF", "FedMD-1r", "FedMD-2r", "FedMD",
           "centralized reference"]
METHODS = [m for m in METHODS if m in set(results.method)]
summary = results.groupby("method").accuracy.mean().reindex(METHODS)
sd = results.groupby(["method", "dataset", "seed"]).accuracy.mean().reset_index()
errors = sd.groupby("method").accuracy.sem().reindex(METHODS)
fig, ax = plt.subplots(figsize=(3.5, 2.1))
ax.bar(range(len(METHODS)), summary, yerr=1.96 * errors, capsize=2,
       color=["#bbbbbb", "#66ccee", "#228833", "#ccbb44", "#4477aa",
              "#ee7733", "#ddaa33", "#aa3377", "#999999"][:len(METHODS)])
ax.set_xticks(range(len(METHODS)), METHODS, fontsize=5.5, rotation=30,
              ha="right")
ax.set_ylabel("test accuracy")
ax.set_ylim(max(0.45, summary.min() - 0.08), 1.0)
ax.grid(axis="y", lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig(FIG / "fig_pattern2.pdf", bbox_inches="tight")
print("fig_pattern2 replotted")
