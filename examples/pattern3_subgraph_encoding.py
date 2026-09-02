#!/usr/bin/env python3
"""Expanded P3 study: scope error, coupling, and classical solver controls.

The deployed state is a feasible local optimum of the pre-event instance, not
the known global optimum. A field disturbance is observed with noise; true,
inferred, and deliberately shifted scopes are compared.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from _common import FIG, RES, plt, write_macros
from sim_core import (anneal, bit_table, channel_instance, channel_milp,
                      costs_from_terms, local_search, tune_qaoa)

SEEDS = 12
N = 16
PATCHES = (4, 6, 8)
COUPLINGS = (0.55, 1.65)
SEVERITIES = (0.65, 1.35)
BUDGET = 89
NOISE_LEVELS = (0.10, 0.25, 0.50)


def states_over_scope(base: int, scope: np.ndarray) -> np.ndarray:
    states = np.full(1 << len(scope), base, dtype=np.int64)
    assignments = bit_table(len(scope))
    for j, qubit in enumerate(scope):
        states &= ~(1 << int(qubit))
        states |= assignments[:, j].astype(np.int64) << int(qubit)
    return states


def infer_scope(delta: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    observed = np.abs(delta + rng.normal(0, 0.45 * np.std(delta) + 1e-6, len(delta)))
    return np.sort(np.argsort(-observed)[:size])


def gap(cost: float, optimum: float) -> float:
    return float((cost - optimum) / (abs(optimum) + 1.0))


def append(rows: list, *, seed: int, patch: int, coupling: float, severity: float,
           scope_name: str, scope: np.ndarray, solver: str, solution: int,
           evals: int, runtime_s: float, costs: np.ndarray, optimum: float,
           p_opt: float = np.nan, depth: int = 0, shots: int = 0,
           boundary_ratio: float = np.nan, noise_lambda: float = 0.0):
    rows.append({
        "seed": seed, "n": N, "patch": patch, "coupling": coupling,
        "severity": severity, "scope": scope_name, "scope_size": len(scope),
        "solver": solver, "solution": solution, "cost": float(costs[solution]),
        "gap": gap(float(costs[solution]), optimum), "evals": evals,
        "runtime_s": runtime_s, "p_opt": p_opt, "depth": depth, "shots": shots,
        "amp_ops": evals * max(1, len(scope)) * (1 << max(1, len(scope))),
        "boundary_ratio": boundary_ratio,
        "noise_lambda": noise_lambda,
    })


def scope_scaling() -> pd.DataFrame:
    """Time one-hop constraint-incidence closure on sparse large graphs."""
    rows = []
    for n in (1_000, 5_000, 10_000):
        for seed in range(20):
            rng = np.random.default_rng(610_000 + n + seed)
            adjacency = [set() for _ in range(n)]
            # A ring prevents isolated variables; two random chords per node
            # approximate a bounded-degree sparse coordination graph.
            for i in range(n):
                for j in ((i + 1) % n, int(rng.integers(n)), int(rng.integers(n))):
                    if i != j:
                        adjacency[i].add(j); adjacency[j].add(i)
            for event in range(10):
                changed = rng.choice(n, size=max(4, n // 1000), replace=False)
                start = time.perf_counter_ns()
                closure = set(map(int, changed))
                for variable in changed:
                    closure.update(adjacency[int(variable)])
                runtime_ms = (time.perf_counter_ns() - start) / 1e6
                rows.append({"n": n, "seed": seed, "event": event,
                             "changed": len(changed), "scope": len(closure),
                             "scope_fraction": len(closure) / n,
                             "runtime_ms": runtime_ms,
                             "edges": sum(map(len, adjacency)) // 2})
    return pd.DataFrame(rows)


def main() -> None:
    rows = []
    for coupling in COUPLINGS:
      for patch in PATCHES:
       for severity in SEVERITIES:
        for seed in range(SEEDS):
            case_start = len(rows)
            rng = np.random.default_rng(70_000 + seed + 100 * patch
                                        + int(1000 * coupling) + int(10 * severity))
            original = channel_instance(N, 20_000 + seed, coupling)
            deployed, _ = local_search(original.costs, N, rng, starts=8)
            start = int(rng.integers(0, N - patch + 1))
            true_scope = np.arange(start, start + patch)
            delta = np.zeros(N)
            delta[true_scope] = rng.normal(0, severity, patch)
            fields = original.fields + delta
            costs = costs_from_terms(fields, original.weights)
            optimum = float(costs.min())
            inferred = infer_scope(delta, patch, rng)
            wrong = np.sort((true_scope + max(1, patch // 2)) % N)
            scopes = {"true": true_scope, "inferred": inferred, "wrong": wrong}

            outside = np.setdiff1d(np.arange(N), true_scope)
            cross = original.weights[np.ix_(true_scope, outside)].sum()
            boundary_ratio = float(cross / (original.weights.sum() + 1e-12))

            append(rows, seed=seed, patch=patch, coupling=coupling,
                   severity=severity, scope_name="full", scope=np.arange(N),
                   solver="deployed", solution=deployed, evals=0, runtime_s=0,
                   costs=costs, optimum=optimum, boundary_ratio=boundary_ratio)

            exact = int(np.argmin(costs))
            append(rows, seed=seed, patch=patch, coupling=coupling,
                   severity=severity, scope_name="full", scope=np.arange(N),
                   solver="exact enumeration", solution=exact, evals=1 << N,
                   runtime_s=0, costs=costs, optimum=optimum,
                   boundary_ratio=boundary_ratio)
            milp_result = channel_milp(fields, original.weights)
            if abs(costs[milp_result["solution"]] - optimum) > 1e-6:
                raise RuntimeError("MILP did not match exact enumeration")
            append(rows, seed=seed, patch=patch, coupling=coupling,
                   severity=severity, scope_name="full", scope=np.arange(N),
                   solver="MILP", solution=milp_result["solution"], evals=0,
                   runtime_s=milp_result["runtime_s"], costs=costs,
                   optimum=optimum, boundary_ratio=boundary_ratio)

            timer = time.perf_counter()
            full_anneal, evaluations = anneal(costs, N, rng, BUDGET, deployed)
            append(rows, seed=seed, patch=patch, coupling=coupling,
                   severity=severity, scope_name="full", scope=np.arange(N),
                   solver="annealing", solution=full_anneal, evals=evaluations,
                   runtime_s=time.perf_counter() - timer, costs=costs,
                   optimum=optimum, boundary_ratio=boundary_ratio)
            qfull = tune_qaoa(costs, N, rng, depth=1, shots=256, max_evals=BUDGET)
            append(rows, seed=seed, patch=patch, coupling=coupling,
                   severity=severity, scope_name="full", scope=np.arange(N),
                   solver="QAOA", solution=qfull["solution"], evals=qfull["evals"],
                   runtime_s=qfull["runtime_s"], costs=costs, optimum=optimum,
                   p_opt=qfull["success_prob"], depth=1, shots=256,
                   boundary_ratio=boundary_ratio)

            for scope_name, scope in scopes.items():
                candidate_states = states_over_scope(deployed, scope)
                reduced_costs = costs[candidate_states]
                timer = time.perf_counter()
                reduced_exact = int(np.argmin(reduced_costs))
                append(rows, seed=seed, patch=patch, coupling=coupling,
                       severity=severity, scope_name=scope_name, scope=scope,
                       solver="exact LNS", solution=int(candidate_states[reduced_exact]),
                       evals=len(reduced_costs), runtime_s=time.perf_counter() - timer,
                       costs=costs, optimum=optimum, boundary_ratio=boundary_ratio)
                timer = time.perf_counter()
                reduced_anneal, evaluations = anneal(reduced_costs, len(scope), rng,
                                                      BUDGET, initial=0)
                append(rows, seed=seed, patch=patch, coupling=coupling,
                       severity=severity, scope_name=scope_name, scope=scope,
                       solver="annealing", solution=int(candidate_states[reduced_anneal]),
                       evals=evaluations, runtime_s=time.perf_counter() - timer,
                       costs=costs, optimum=optimum, boundary_ratio=boundary_ratio)
                qpatch = tune_qaoa(reduced_costs, len(scope), rng, depth=1,
                                   shots=256, max_evals=BUDGET)
                append(rows, seed=seed, patch=patch, coupling=coupling,
                       severity=severity, scope_name=scope_name, scope=scope,
                       solver="QAOA", solution=int(candidate_states[qpatch["solution"]]),
                       evals=qpatch["evals"], runtime_s=qpatch["runtime_s"],
                       costs=costs, optimum=optimum, p_opt=qpatch["success_prob"],
                       depth=1, shots=256, boundary_ratio=boundary_ratio)

            # Depth/shot sensitivity is run on the inferred operational scope.
            reduced_states = states_over_scope(deployed, inferred)
            reduced_costs = costs[reduced_states]
            for depth, shots in ((1, 64), (2, 64), (2, 256), (2, 1024)):
                qpatch = tune_qaoa(reduced_costs, len(inferred), rng, depth=depth,
                                   shots=shots, max_evals=BUDGET)
                append(rows, seed=seed, patch=patch, coupling=coupling,
                       severity=severity, scope_name="inferred-sensitivity", scope=inferred,
                       solver="QAOA", solution=int(reduced_states[qpatch["solution"]]),
                       evals=qpatch["evals"], runtime_s=qpatch["runtime_s"],
                       costs=costs, optimum=optimum, p_opt=qpatch["success_prob"],
                       depth=depth, shots=shots, boundary_ratio=boundary_ratio)
            # Fidelity rung f2: the final QAOA distribution is mixed with a
            # uniform distribution by a declared global depolarizing channel.
            # It is an ablation, not a device-calibrated noise claim.
            for noise_lambda in NOISE_LEVELS:
                qpatch = tune_qaoa(reduced_costs, len(inferred), rng, depth=1,
                                   shots=256, max_evals=BUDGET,
                                   noise_lambda=noise_lambda)
                append(rows, seed=seed, patch=patch, coupling=coupling,
                       severity=severity, scope_name="inferred-noise", scope=inferred,
                       solver="QAOA", solution=int(reduced_states[qpatch["solution"]]),
                       evals=qpatch["evals"], runtime_s=qpatch["runtime_s"],
                       costs=costs, optimum=optimum, p_opt=qpatch["success_prob"],
                       depth=1, shots=256, boundary_ratio=boundary_ratio,
                       noise_lambda=noise_lambda)
            recall = {
                "full": 1.0, "true": 1.0,
                "inferred": len(set(inferred) & set(true_scope)) / patch,
                "inferred-sensitivity": len(set(inferred) & set(true_scope)) / patch,
                "inferred-noise": len(set(inferred) & set(true_scope)) / patch,
                "wrong": len(set(wrong) & set(true_scope)) / patch,
            }
            for row in rows[case_start:]:
                row["scope_recall"] = recall[row["scope"]]
            print(f"P3 c={coupling:.2f} k={patch} severity={severity:.2f} seed={seed}",
                  flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RES / "pattern3_subgraph.csv", index=False)
    scaling = scope_scaling()
    scaling.to_csv(RES / "pattern3_scope_scale.csv", index=False)

    main = frame[(frame.severity == SEVERITIES[1]) &
                 (((frame.scope == "full") & frame.solver.isin(["deployed", "annealing", "QAOA", "MILP"])) |
                  ((frame.scope.isin(["true", "inferred", "wrong"])) &
                   frame.solver.isin(["exact LNS", "annealing", "QAOA"])))]
    display = [("full", "deployed"), ("full", "QAOA"),
               ("inferred", "QAOA"), ("inferred", "annealing"),
               ("inferred", "exact LNS"), ("full", "MILP")]
    means = [main[(main.scope == s) & (main.solver == a)].gap.mean() for s, a in display]
    seedmeans = [main[(main.scope == s) & (main.solver == a)].groupby("seed").gap.mean()
                 for s, a in display]
    errors = [1.96 * x.sem() for x in seedmeans]
    labels = ["deployed", "full\nQAOA", "inferred\nQAOA", "inferred\nanneal",
              "exact\nLNS", "MILP"]
    fig, ax = plt.subplots(figsize=(3.5, 2.15))
    ax.bar(range(len(display)), means, yerr=errors, capsize=2,
           color=["#bbbbbb", "#ee6677", "#4477aa", "#ccbb44", "#228833", "#888888"])
    ax.set_xticks(range(len(display)), labels, fontsize=6)
    ax.set_ylabel("normalized post-event gap")
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_pattern3.pdf", bbox_inches="tight")

    reference = frame[(frame.patch == 6) & (frame.scope == "inferred") &
                      (frame.solver == "QAOA") & (frame.depth == 1) &
                      (frame.shots == 256) & (frame.noise_lambda == 0)]
    reference_lns = frame[(frame.patch == 6) & (frame.scope == "inferred") &
                          (frame.solver == "exact LNS")]
    full = frame[(frame.patch == 6) & (frame.scope == "full") &
                 (frame.solver == "QAOA")]
    full_milp = frame[(frame.patch == 6) & (frame.scope == "full") &
                      (frame.solver == "MILP")]
    analytic_ratio = (N / 6) * (2 ** (N - 6))
    noise = frame[(frame.patch == 6) & (frame.scope == "inferred-noise") &
                  (frame.solver == "QAOA") & (frame.depth == 1) &
                  (frame.shots == 256)]
    noise_gap = noise.groupby("noise_lambda").gap.mean()
    noise_popt = noise.groupby("noise_lambda").p_opt.mean()
    ideal_popt = float(reference.p_opt.mean())
    shots95 = lambda probability: int(np.ceil(np.log(0.05) / np.log(1 - probability)))
    scale_10k = scaling[scaling.n == 10_000]
    write_macros("pattern3", {
        "PatThreeSeeds": SEEDS, "PatThreePatchMin": min(PATCHES),
        "PatThreePatchMax": max(PATCHES), "PatThreeConditions": len(PATCHES) * len(COUPLINGS) * len(SEVERITIES),
        "PatThreeInferredGap": f"{reference.gap.mean():.3f}",
        "PatThreeFullGap": f"{full.gap.mean():.3f}",
        "PatThreeOpsFormula": f"{analytic_ratio:.0f}",
        "PatThreePOptFull": f"{100 * full.p_opt.mean():.2f}\\%",
        "PatThreePOptPatch": f"{100 * reference.p_opt.mean():.1f}\\%",
        "PatThreeScopeRecall": f"{100 * reference.scope_recall.mean():.1f}\\%",
        "PatThreeMILPGap": f"{frame[(frame.scope == 'full') & (frame.solver == 'MILP')].gap.mean():.3f}",
        "PatThreePatchQaoaMs": f"{1000 * reference.runtime_s.mean():.2f}",
        "PatThreePatchLnsMs": f"{1000 * reference_lns.runtime_s.mean():.3f}",
        "PatThreeFullQaoaMs": f"{1000 * full.runtime_s.mean():.1f}",
        "PatThreeFullMilpMs": f"{1000 * full_milp.runtime_s.mean():.1f}",
        "PatThreeNoiseMid": f"{NOISE_LEVELS[1]:.2f}",
        "PatThreeNoiseGap": f"{noise_gap.loc[NOISE_LEVELS[1]]:.3f}",
        "PatThreeNoisePOpt": f"{100 * noise_popt.loc[NOISE_LEVELS[1]]:.1f}\\%",
        "PatThreeNoiseHighGap": f"{noise_gap.loc[NOISE_LEVELS[-1]]:.3f}",
        "PatThreeNoiseHighPOpt": f"{100 * noise_popt.loc[NOISE_LEVELS[-1]]:.1f}\\%",
        "PatThreeIdealShotsNinetyFive": shots95(ideal_popt),
        "PatThreeNoiseShotsNinetyFive": shots95(float(noise_popt.loc[NOISE_LEVELS[-1]])),
        "PatThreeScopeTenKMedianMs": f"{scale_10k.runtime_ms.median():.3f}",
        "PatThreeScopeTenKTailMs": f"{scale_10k.runtime_ms.quantile(0.95):.3f}",
        "PatThreeScopeTenKFraction": f"{100 * scale_10k.scope_fraction.mean():.2f}\\%",
    })
    print(main.groupby(["scope", "solver"]).gap.mean().round(4))


if __name__ == "__main__":
    main()
