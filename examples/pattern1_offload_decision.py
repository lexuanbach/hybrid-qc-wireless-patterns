#!/usr/bin/env python3
"""Expanded P1 study: contextual execution under drift and link outages.

Eight calibration workload seeds are separated from 24 reporting seeds. The
reporting unit for uncertainty is the workload seed, not an individual context.
All action outcomes are modeled; no row is a hardware observation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from _common import FIG, RES, plt, write_macros

ACTIONS = ("invoke-QPU", "remote-sim", "local-sim", "fallback")
SCALES = (12, 20)
TRAIN_SEEDS = tuple(range(8))
TEST_SEEDS = tuple(range(8, 32))
CONTEXTS = 30
COMPONENTS = ("quality_loss", "delay_s", "deadline_miss", "infeasible",
              "energy_j", "money")
FEATURES = ("difficulty", "queue_s", "rtt_s", "fail_prob", "deadline_s",
            "volatility", "reuse_count", "qpu_error", "outage", "n")


@dataclass(frozen=True)
class Scenario:
    name: str = "reference"
    outage_p: float = 0.30
    deadline_scale: float = 1.0
    queue_scale: float = 1.0
    error_scale: float = 1.0
    shots: int = 512
    reuse_scale: float = 1.0
    w_deadline: float = 2.0
    w_infeasible: float = 5.0
    w_money: float = 0.5
    w_energy: float = 0.3
    money_scale: float = 0.01
    energy_scale: float = 50.0


REFERENCE = Scenario()


def utility(row: dict | pd.Series, scenario: Scenario) -> float:
    stale = float(row["volatility"]) * float(row["delay_s"])
    miss = float(row.get("deadline_miss",
                         float(row["delay_s"]) > float(row["deadline_s"])))
    return -(float(row["quality_loss"]) + stale
             + scenario.w_deadline * miss
             + scenario.w_infeasible * float(row["infeasible"])
             + scenario.w_money * float(row["money"]) / scenario.money_scale
             + scenario.w_energy * float(row["energy_j"]) / scenario.energy_scale)


def sample_context(n: int, seed: int, index: int, scenario: Scenario) -> dict:
    rng = np.random.default_rng(10_000 * seed + index)
    outage = bool(rng.random() < scenario.outage_p)
    rtt = float(np.exp(rng.normal(np.log(0.035), 0.45)))
    fail = float(rng.uniform(1e-4, 4e-3))
    if outage:
        rtt = min(20.0, rtt * rng.uniform(25, 80))
        fail = min(0.55, fail * rng.uniform(80, 180))
    return {
        "n": n,
        "difficulty": float(rng.uniform(0.65, 1.35)),
        "queue_s": float(np.exp(rng.normal(np.log(0.40), 0.85)) * scenario.queue_scale),
        "rtt_s": rtt,
        "fail_prob": fail,
        "deadline_s": float(np.exp(rng.normal(np.log(1.5), 0.75))
                            * scenario.deadline_scale),
        "volatility": float(rng.uniform(0.01, 0.12)),
        "reuse_count": max(1, int(np.exp(rng.uniform(0, 7)) * scenario.reuse_scale)),
        "qpu_error": float(np.clip(np.exp(rng.normal(np.log(0.012), 0.45))
                                    * scenario.error_scale, 0.001, 0.12)),
        "outage": int(outage),
    }


def action_components(context: dict, action: str, scenario: Scenario,
                      seed: int, index: int) -> dict:
    """Draw one held-out modeled outcome for a complete action."""
    rng = np.random.default_rng(1_000_000 * seed + 1_000 * index + ACTIONS.index(action))
    n, d = int(context["n"]), float(context["difficulty"])
    shots, reuse = scenario.shots, context["reuse_count"]
    if action == "invoke-QPU":
        quality = 0.015 + 1.8 * context["qpu_error"] * d + 0.38 / np.sqrt(shots)
        delay = context["queue_s"] + 3 * context["rtt_s"] + 0.008 + 1.0e-4 * shots + 0.20 / reuse
        energy, money = 5.0 + 0.001 * shots, 1.0e-6 * shots
        failed = rng.random() < 1 - (1 - context["fail_prob"]) ** 3
    elif action == "remote-sim":
        quality = 0.047 + 0.018 * d + rng.normal(0, 0.004)
        delay = 2 * context["rtt_s"] + n * 2 ** n / 2.0e9 + 0.05 / reuse
        energy, money = 250.0 * delay, 1.0e-5 * delay
        failed = rng.random() < 1 - (1 - context["fail_prob"]) ** 2
    elif action == "local-sim":
        quality = 0.082 + 0.012 * d + rng.normal(0, 0.004)
        delay = n * 2 ** n / 5.0e5 + 0.08 / reuse
        energy, money, failed = 8.0 * delay, 0.0, False
    else:
        quality = 0.155 + 0.045 * d + rng.normal(0, 0.006)
        delay, energy, money, failed = 0.025, 0.20, 0.0, False
    row = dict(context)
    row.update({"action": action, "quality_loss": max(0.0, float(quality)),
                "delay_s": float(delay), "infeasible": int(failed),
                "energy_j": float(energy), "money": float(money)})
    row["deadline_miss"] = int(row["delay_s"] > row["deadline_s"])
    row["utility"] = utility(row, scenario)
    return row


def generate(seeds: tuple[int, ...], scenario: Scenario) -> pd.DataFrame:
    rows = []
    for n in SCALES:
        for seed in seeds:
            for index in range(CONTEXTS):
                context = sample_context(n, seed, index, scenario)
                for action in ACTIONS:
                    rows.append({"seed": seed, "context": index,
                                 **action_components(context, action, scenario, seed, index)})
    return pd.DataFrame(rows)


def behavior_log(training: pd.DataFrame) -> pd.DataFrame:
    """Retain one explored action and its propensity per training context."""
    probabilities = np.array([0.20, 0.15, 0.55, 0.10])
    rows = []
    for (n, seed, context), group in training.groupby(["n", "seed", "context"]):
        rng = np.random.default_rng(730_000 + 1000 * int(n)
                                    + 31 * int(seed) + int(context))
        action = str(rng.choice(ACTIONS, p=probabilities))
        row = group[group.action == action].iloc[0].copy()
        row["propensity"] = probabilities[ACTIONS.index(action)]
        rows.append(row)
    return pd.DataFrame(rows)


def fit_component_models(training: pd.DataFrame,
                         propensity_weighted: bool = False) -> dict:
    models = {}
    for n in SCALES:
        for action in ACTIONS:
            part = training[(training.n == n) & (training.action == action)]
            model = RandomForestRegressor(n_estimators=48, max_depth=8,
                                          min_samples_leaf=3, random_state=n + len(action),
                                          n_jobs=-1)
            weights = (1.0 / part.propensity.to_numpy()
                       if propensity_weighted else None)
            model.fit(part[list(FEATURES)], part[list(COMPONENTS)],
                      sample_weight=weights)
            models[n, action] = model
    return models


def attach_predictions(reporting: pd.DataFrame, models: dict,
                       scenario: Scenario, prefix: str = "pred_") -> pd.DataFrame:
    reporting = reporting.copy()
    for component in COMPONENTS:
        reporting[f"{prefix}{component}"] = np.nan
    for n in SCALES:
        for action in ACTIONS:
            mask = (reporting.n == n) & (reporting.action == action)
            values = models[n, action].predict(reporting.loc[mask, list(FEATURES)])
            for j, component in enumerate(COMPONENTS):
                reporting.loc[mask, f"{prefix}{component}"] = values[:, j]
    reporting[f"{prefix}infeasible"] = reporting[f"{prefix}infeasible"].clip(0, 1)
    reporting[f"{prefix}deadline_miss"] = reporting[
        f"{prefix}deadline_miss"].clip(0, 1)
    return reporting


def predicted_table(context_rows: pd.DataFrame, scenario: Scenario,
                    prefix: str = "pred_") -> dict[str, dict]:
    predicted = {}
    for action in ACTIONS:
        row = context_rows[context_rows.action == action].iloc[0]
        comp = {component: float(row[f"{prefix}{component}"])
                for component in COMPONENTS}
        comp.update({k: float(row[k]) for k in ("deadline_s", "volatility")})
        predicted[action] = comp
        predicted[action]["utility"] = utility(comp, scenario)
    return predicted


def pareto_action(predicted: dict[str, dict]) -> str:
    fields = ("quality_loss", "delay_s", "infeasible", "energy_j", "money")
    matrix = np.array([[predicted[a][f] for f in fields] for a in ACTIONS])
    keep = []
    for i in range(len(ACTIONS)):
        dominated = any(np.all(matrix[j] <= matrix[i]) and np.any(matrix[j] < matrix[i])
                        for j in range(len(ACTIONS)) if j != i)
        if not dominated:
            keep.append(i)
    lo, hi = matrix.min(axis=0), matrix.max(axis=0)
    normalized = (matrix - lo) / (hi - lo + 1e-12)
    return ACTIONS[min(keep, key=lambda i: float(normalized[i].max()))]


class LinUCB:
    def __init__(self, training: pd.DataFrame, alpha: float = 0.35):
        self.alpha = alpha
        self.A = {a: np.eye(len(FEATURES) + 1) for a in ACTIONS}
        self.b = {a: np.zeros(len(FEATURES) + 1) for a in ACTIONS}
        for row in training.itertuples():
            x = self.features(row)
            self.A[row.action] += np.outer(x, x)
            self.b[row.action] += x * row.utility

    @staticmethod
    def features(row) -> np.ndarray:
        raw = np.array([getattr(row, k) for k in FEATURES], dtype=float)
        scales = np.array([1.0, 10.0, 10.0, 0.5, 10.0, 0.2, 1000.0, 0.1, 1.0, 20.0])
        return np.concatenate([[1.0], np.clip(raw / scales, -5, 5)])

    def choose(self, rows: pd.DataFrame) -> str:
        scores = {}
        for row in rows.itertuples():
            x = self.features(row)
            inv = np.linalg.inv(self.A[row.action])
            scores[row.action] = float(x @ inv @ self.b[row.action]
                                       + self.alpha * np.sqrt(x @ inv @ x))
        return max(scores, key=scores.get)

    def update(self, row: pd.Series) -> None:
        class Obj: pass
        obj = Obj()
        for key in FEATURES:
            setattr(obj, key, row[key])
        x = self.features(obj)
        action = str(row.action)
        self.A[action] += np.outer(x, x)
        self.b[action] += x * float(row.utility)


def evaluate_policies(training: pd.DataFrame, reporting: pd.DataFrame,
                      scenario: Scenario) -> tuple[pd.DataFrame, dict]:
    models = fit_component_models(training)
    reporting = attach_predictions(reporting, models, scenario)
    logged = behavior_log(training)
    logged_models = fit_component_models(logged, propensity_weighted=True)
    reporting = attach_predictions(reporting, logged_models, scenario, "ipw_")
    train_fixed = training.groupby(["n", "action"]).utility.mean()
    selected_fixed = {n: train_fixed[n].idxmax() for n in SCALES}
    bandits = {n: LinUCB(training[training.n == n]) for n in SCALES}
    rows = []
    for (n, seed, context), group in reporting.groupby(["n", "seed", "context"], sort=True):
        predicted = predicted_table(group, scenario)
        ipw_predicted = predicted_table(group, scenario, "ipw_")
        admissible = [a for a in ACTIONS
                      if predicted[a]["deadline_miss"] <= 0.10
                      and predicted[a]["infeasible"] <= 0.10]
        choices = {
            "P1": max(admissible or ["fallback"],
                      key=lambda a: predicted[a]["utility"]),
            "utility-only": max(predicted,
                                key=lambda a: predicted[a]["utility"]),
            "logged-IPW": max(ipw_predicted,
                              key=lambda a: ipw_predicted[a]["utility"]),
            "LinUCB": bandits[n].choose(group),
            "Pareto": pareto_action(predicted),
            "oracle": group.loc[group.utility.idxmax()].action,
            "train-selected fixed": selected_fixed[n],
        }
        choices.update({f"always {a}": a for a in ACTIONS})
        for policy, action in choices.items():
            outcome = group[group.action == action].iloc[0]
            rows.append({"n": n, "seed": seed, "context": context,
                         "policy": policy, "action": action,
                         "utility": outcome.utility,
                         "deadline_miss": outcome.deadline_miss,
                         "infeasible": outcome.infeasible})
        bandits[n].update(group[group.action == choices["LinUCB"]].iloc[0])
    return pd.DataFrame(rows), selected_fixed


def sensitivity_grid() -> pd.DataFrame:
    scenarios = [
        replace(REFERENCE, name=f"outage={x}", outage_p=x) for x in (0.05, 0.15, 0.30, 0.50)
    ] + [
        replace(REFERENCE, name=f"deadline={x}", deadline_scale=x) for x in (0.5, 1.0, 2.0)
    ] + [
        replace(REFERENCE, name=f"queue={x}", queue_scale=x) for x in (0.5, 1.0, 2.0)
    ] + [
        replace(REFERENCE, name=f"error={x}", error_scale=x) for x in (0.5, 1.0, 2.0)
    ] + [
        replace(REFERENCE, name=f"shots={x}", shots=x) for x in (128, 512, 2048)
    ] + [
        replace(REFERENCE, name=f"reuse={x}", reuse_scale=x) for x in (0.25, 1.0, 4.0)
    ] + [
        replace(REFERENCE, name="latency-priority", w_deadline=4.0, w_infeasible=8.0),
        replace(REFERENCE, name="cost-priority", w_money=1.5, w_energy=1.0),
        replace(REFERENCE, name="quality-priority", w_deadline=0.5, w_money=0.1, w_energy=0.1),
        replace(REFERENCE, name="normalizers-tight", money_scale=0.005, energy_scale=25.0),
        replace(REFERENCE, name="normalizers-loose", money_scale=0.02, energy_scale=100.0),
    ]
    rows = []
    # Eight training and 20 reporting seeds keep every sensitivity estimate
    # clusterable without making contexts the independent units.
    for scenario in scenarios:
        train = generate(tuple(range(8)), scenario)
        test = generate(tuple(range(8, 28)), scenario)
        evaluated, _ = evaluate_policies(train, test, scenario)
        part = evaluated[evaluated.policy.isin(["P1", "train-selected fixed", "oracle"])]
        summary = part.groupby(["n", "policy"]).agg(
            utility=("utility", "mean"), miss=("deadline_miss", "mean"),
            infeasible=("infeasible", "mean")).reset_index()
        summary["scenario"] = scenario.name
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    training = generate(TRAIN_SEEDS, REFERENCE)
    reporting = generate(TEST_SEEDS, REFERENCE)
    evaluated, selected_fixed = evaluate_policies(training, reporting, REFERENCE)
    reporting.to_csv(RES / "pattern1_all_actions.csv", index=False)
    evaluated.to_csv(RES / "pattern1_offload.csv", index=False)
    sensitivity = sensitivity_grid()
    sensitivity.to_csv(RES / "pattern1_sensitivity.csv", index=False)
    (RES / "pattern1_config.json").write_text(json.dumps(REFERENCE.__dict__, indent=2) + "\n")

    policies = ["P1", "utility-only", "logged-IPW", "LinUCB", "Pareto", "oracle",
                "train-selected fixed"]
    means = evaluated[evaluated.policy.isin(policies)].pivot_table(
        index="policy", columns="n", values="utility", aggfunc="mean").reindex(policies)
    seed_means = evaluated[evaluated.policy.isin(policies)].groupby(
        ["n", "seed", "policy"]).utility.mean().reset_index()
    sem = seed_means.pivot_table(index="policy", columns="n", values="utility",
                                 aggfunc="sem").reindex(policies)
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    x = np.arange(len(policies))
    for j, n in enumerate(SCALES):
        ax.bar(x + (j - 0.5) * 0.38, means[n], 0.36, yerr=1.96 * sem[n],
               capsize=2, label=f"$n={n}$", color=("#4477aa", "#66ccee")[j])
    ax.set_xticks(x, [p.replace("-", "-\n", 1) for p in policies], fontsize=5.5)
    ax.set_ylabel("mean realized utility")
    ax.set_yscale("symlog", linthresh=0.08)
    ax.legend(frameon=False, ncol=2, loc="lower left",
              borderaxespad=0.2, handlelength=1.2)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_pattern1.pdf", bbox_inches="tight")

    action_mix = evaluated[evaluated.policy == "P1"].action.value_counts(normalize=True)
    paired = evaluated.pivot_table(index=["n", "seed", "context"],
                                   columns="policy", values="utility")
    write_macros("pattern1", {
        "PatOneTrainSeeds": len(TRAIN_SEEDS), "PatOneTestSeeds": len(TEST_SEEDS),
        "PatOneContexts": len(TEST_SEEDS) * CONTEXTS,
        "PatOneCalSmall": f"{means.loc['P1', 12]:.3f}",
        "PatOneCalLarge": f"{means.loc['P1', 20]:.3f}",
        "PatOneCalSmallPrecise": f"{means.loc['P1', 12]:.4f}",
        "PatOneFixedSmallPrecise": f"{means.loc['train-selected fixed', 12]:.4f}",
        "PatOneOracleSmallPrecise": f"{means.loc['oracle', 12]:.4f}",
        "PatOnePairSmallPrecise": f"{(paired.loc[12, 'P1'] - paired.loc[12, 'train-selected fixed']).mean():+.4f}",
        "PatOneBanditSmall": f"{means.loc['LinUCB', 12]:.3f}",
        "PatOneBanditLarge": f"{means.loc['LinUCB', 20]:.3f}",
        "PatOneIpwSmall": f"{means.loc['logged-IPW', 12]:.3f}",
        "PatOneIpwLarge": f"{means.loc['logged-IPW', 20]:.3f}",
        "PatOneOracleSmall": f"{means.loc['oracle', 12]:.3f}",
        "PatOneOracleLarge": f"{means.loc['oracle', 20]:.3f}",
        "PatOneFixedSmall": f"{means.loc['train-selected fixed', 12]:.3f}",
        "PatOneFixedLarge": f"{means.loc['train-selected fixed', 20]:.3f}",
        "PatOneQpuShare": f"{100 * action_mix.get('invoke-QPU', 0):.0f}\\%",
        "PatOneRemoteShare": f"{100 * action_mix.get('remote-sim', 0):.0f}\\%",
        "PatOneLocalShare": f"{100 * action_mix.get('local-sim', 0):.0f}\\%",
        "PatOneFallbackShare": f"{100 * action_mix.get('fallback', 0):.0f}\\%",
        "PatOneSensitivityCases": sensitivity.scenario.nunique(),
    })
    print("training-selected fixed:", selected_fixed)
    print(means.round(3))


if __name__ == "__main__":
    main()
