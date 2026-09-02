#!/usr/bin/env python3
"""Regenerate fig_pattern1.pdf from stored results (layout-only changes;
numbers and macros untouched). Compact vertical profile for the column."""
import numpy as np
import pandas as pd
from _common import RES, FIG, plt

d = pd.read_csv(RES / "pattern1_offload.csv")
policies = [p for p in ["P1", "logged-IPW", "LinUCB", "utility-only",
            "Pareto", "oracle", "train-selected fixed"]
            if p in set(pd.read_csv(RES / "pattern1_offload.csv").policy)]
labels = {"P1": "P1\n(calibrated)", "train-selected fixed": "train-\nselected fixed"}
d = d[d.policy.isin(policies)]
means = d.pivot_table(index="policy", columns="n", values="utility",
                      aggfunc="mean").reindex(policies)
seed_means = d.groupby(["n", "seed", "policy"]).utility.mean().reset_index()
sem = seed_means.pivot_table(index="policy", columns="n", values="utility",
                             aggfunc="sem").reindex(policies)
fig, ax = plt.subplots(figsize=(3.5, 1.75))
x = np.arange(len(policies))
for j, n in enumerate((12, 20)):
    ax.bar(x + (j - 0.5) * 0.38, means[n], 0.36, yerr=1.96 * sem[n],
           capsize=2, label=f"$n={n}$", color=("#4477aa", "#66ccee")[j])
ax.set_xticks(x, [labels.get(p, p.replace("-", "-\n", 1)) for p in policies],
              fontsize=5.5)
ax.set_ylabel("mean realized\nutility")
# lower linthresh gives the shallow bars usable resolution and removes the
# empty mid-band the previous linthresh created
ax.set_yscale("symlog", linthresh=0.08)
ax.legend(frameon=False, ncol=2, loc="lower left", borderaxespad=0.2,
          handlelength=1.2, columnspacing=0.8)
ax.grid(axis="y", lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig(FIG / "fig_pattern1.pdf", bbox_inches="tight")
print("fig_pattern1.pdf replotted (compact)")
