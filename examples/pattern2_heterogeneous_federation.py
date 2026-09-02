#!/usr/bin/env python3
"""Expanded P2 study with FedMD/FedDF, soft targets, and diagnostics.

Public labels are carried only for assertions and final bookkeeping; no
training, weighting, calibration, or model-selection path reads them.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from _common import FIG, RES, plt, write_macros
from sim_core import vqc_forward

SEEDS = 12
ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cancer", "digits", "wine", "ran-5g")
METHODS = ("best local", "unweighted ensemble", "weighted ensemble",
           "hard transfer", "FedDF", "FedMD-1r", "FedMD-2r", "FedMD",
           "centralized reference")


@dataclass(frozen=True)
class Regime:
    public_fraction: float = 0.24
    noniid: str = "severe"
    noise_lambda: float = 0.55
    shots: int = 64
    clients: int = 3
    rounds: int = 3
    vqc_iters: int = 45


MAIN = Regime()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def raw_dataset(name: str):
    if name == "cancer":
        data = load_breast_cancer(); X, y = data.data, data.target
    elif name == "digits":
        data = load_digits(); mask = np.isin(data.target, [3, 8])
        X, y = data.data[mask], (data.target[mask] == 8).astype(int)
    elif name == "wine":
        data = load_wine(); mask = data.target < 2
        X, y = data.data[mask], data.target[mask]
    elif name == "ran-5g":
        # Each sample predicts whether the same measured cell exceeds 20% PRB
        # use in the next half-hour.  Features use only the current and previous
        # window; the fixed engineering threshold avoids label-derived scaling.
        data = pd.read_csv(ROOT / "data" / "netdata_5g_cells.csv")
        data = data.sort_values(["Cell ID", "minute"]).copy()
        data["lag_prb"] = data.groupby("Cell ID").prb_pct.shift(1)
        data["lag_traffic"] = data.groupby("Cell ID").traffic_kbyte.shift(1)
        data["next_prb"] = data.groupby("Cell ID").prb_pct.shift(-1)
        data = data.dropna().copy()
        angle = 2 * np.pi * data.minute.to_numpy() / 1440.0
        numeric = np.column_stack([
            data.prb_pct, np.log1p(data.traffic_kbyte), data.users,
            data.lag_prb, np.log1p(data.lag_traffic),
            np.sin(angle), np.cos(angle),
        ])
        cell = pd.get_dummies(data["Cell ID"], dtype=float).to_numpy()
        station = pd.get_dummies(data["Base Station ID"].astype(str),
                                 dtype=float).to_numpy()
        X = np.column_stack([numeric, cell, station])
        y = (data.next_prb.to_numpy() >= 20.0).astype(int)
    else:
        raise KeyError(name)
    return np.asarray(X, float), np.asarray(y, int)


def partition(name: str, seed: int, regime: Regime):
    X, y = raw_dataset(name)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.24, stratify=y, random_state=1000 + seed)
    n_public = max(24, int(round(regime.public_fraction * len(X))))
    n_public = min(n_public, len(X_dev) - 2 * regime.clients)
    X_private, X_public, y_private, y_public = train_test_split(
        X_dev, y_dev, test_size=n_public, stratify=y_dev, random_state=2000 + seed)
    # Unsupervised PCA/scaling is fitted on the public feature pool only.
    dims = min(6, X_public.shape[1], len(X_public) - 1)
    pca = PCA(n_components=dims, random_state=seed).fit(X_public)
    X_public = pca.transform(X_public); X_private = pca.transform(X_private)
    X_test = pca.transform(X_test)
    mu, sd = X_public.mean(0), X_public.std(0) + 1e-9
    transform = lambda z: np.clip((z - mu) / sd, -2.5, 2.5)
    return transform(X_private), y_private, transform(X_public), y_public, transform(X_test), y_test


def make_shards(X: np.ndarray, y: np.ndarray, clients: int, noniid: str,
                seed: int):
    rng = np.random.default_rng(seed)
    if noniid == "iid":
        order = rng.permutation(len(X))
    elif noniid == "moderate":
        order = np.argsort(X[:, 0] + rng.normal(0, 0.8, len(X)))
    else:
        order = np.argsort(X[:, 0])
    parts = [list(p) for p in np.array_split(order, clients)]
    # Deterministic class guard; moved samples stay private and disjoint.
    for i in range(clients):
        for cls in (0, 1):
            if not any(y[j] == cls for j in parts[i]):
                donor = max(range(clients), key=lambda k: sum(y[j] == cls for j in parts[k]))
                candidate = next(j for j in parts[donor] if y[j] == cls)
                parts[donor].remove(candidate); parts[i].append(candidate)
    return [(X[np.array(indices)], y[np.array(indices)]) for indices in parts]


def fit_soft_head(features: np.ndarray, targets: np.ndarray,
                  sample_weight: np.ndarray | None = None,
                  initial: np.ndarray | None = None) -> np.ndarray:
    targets = np.asarray(targets, float)
    Z = np.column_stack([np.ones(len(features)), features])
    weights = np.ones(len(Z)) if sample_weight is None else np.asarray(sample_weight, float)

    def objective(beta):
        p = sigmoid(Z @ beta)
        ce = -(targets * np.log(p + 1e-10) + (1 - targets) * np.log(1 - p + 1e-10))
        return float(np.average(ce, weights=weights) + 2e-3 * np.dot(beta[1:], beta[1:]))

    start = np.zeros(Z.shape[1]) if initial is None else initial
    return minimize(objective, start, method="L-BFGS-B", options={"maxiter": 120}).x


def predict_head(beta: np.ndarray, features: np.ndarray) -> np.ndarray:
    return sigmoid(np.column_stack([np.ones(len(features)), features]) @ beta)


def train_representation(kind: str, X: np.ndarray, y: np.ndarray, seed: int,
                         iters: int):
    if kind == "classical":
        return {"kind": kind, "transform": lambda z: z,
                "private_features": X}
    n = 4
    rng = np.random.default_rng(seed)
    theta0 = rng.normal(0, 0.35, 2 * n)

    def objective(theta):
        features = vqc_forward(X[:, :n], theta, n=n, layers=2)
        beta = fit_soft_head(features, y)
        return float(np.mean((predict_head(beta, features) - y) ** 2))

    theta = minimize(objective, theta0, method="COBYLA",
                     options={"maxiter": iters}).x
    return {"kind": kind, "theta": theta,
            "transform": lambda z, th=theta: vqc_forward(z[:, :n], th, n=n, layers=2),
            "private_features": vqc_forward(X[:, :n], theta, n=n, layers=2)}


def noisy(features: np.ndarray, lam: float, shots: int,
          rng: np.random.Generator) -> np.ndarray:
    degraded = (1 - lam) * features
    sigma = np.sqrt(np.clip(1 - degraded ** 2, 0.05, 1) / shots)
    return degraded + rng.normal(0, sigma)


def ece(probability: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probability >= lo) & (probability < hi if hi < 1 else probability <= hi)
        if np.any(mask):
            total += mask.mean() * abs(probability[mask].mean() - y[mask].mean())
    return float(total)


def metrics(probability: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    return (float(np.mean((probability >= 0.5) == y)),
            float(np.mean((probability - y) ** 2)), ece(probability, y))


def parameter_bytes(client: dict) -> int:
    total = int(client["beta"].nbytes)
    if "theta" in client:
        total += int(client["theta"].nbytes)
    return total


def median_runtime_ms(function, repeats: int = 9) -> float:
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return float(np.median(values))


def run_once(dataset_name: str, seed: int, regime: Regime) -> list[dict]:
    X_private, y_private, X_public, y_public, X_test, y_test = partition(
        dataset_name, seed, regime)
    shards = make_shards(X_private, y_private, regime.clients, regime.noniid, seed)
    kinds = ["classical", "sim_vqc", "qpu_vqc"]
    clients = []
    for index, (X_local, y_local) in enumerate(shards):
        kind = kinds[index % len(kinds)]
        rep = train_representation(kind, X_local, y_local, 5000 + 31 * seed + index,
                                   regime.vqc_iters)
        private_features = rep["private_features"]
        public_features = rep["transform"](X_public)
        test_features = rep["transform"](X_test)
        own_features = private_features
        if kind == "qpu_vqc":
            private_features = noisy(private_features, regime.noise_lambda, regime.shots,
                                     np.random.default_rng(100 + seed + index))
            public_features = noisy(public_features, regime.noise_lambda, regime.shots,
                                    np.random.default_rng(200 + seed + index))
            test_features = noisy(test_features, regime.noise_lambda, regime.shots,
                                  np.random.default_rng(300 + seed + index))
        beta = fit_soft_head(private_features, y_local)
        clients.append({"kind": kind, "X": X_local, "y": y_local,
                        "private_features": private_features,
                        "public_features": public_features,
                        "test_features": test_features, "beta": beta,
                        "own_features": own_features,
                        "transform": rep["transform"],
                        **({"theta": rep["theta"]} if "theta" in rep else {})})

    public_predictions = np.vstack([predict_head(c["beta"], c["public_features"])
                                    for c in clients])
    test_predictions = np.vstack([predict_head(c["beta"], c["test_features"])
                                  for c in clients])
    local_scores = np.array([metrics(p, y_test)[0] for p in test_predictions])
    best_local = test_predictions[int(np.argmax(local_scores))]
    ensemble = test_predictions.mean(0)
    consensus = public_predictions.mean(0)
    disagreement = np.mean((public_predictions - consensus[None, :]) ** 2, axis=1)
    reliability = 1 / (disagreement + 1e-3); reliability /= reliability.sum()
    weighted_public = reliability @ public_predictions
    weighted_test = reliability @ test_predictions

    hard_beta = fit_soft_head(X_public, (consensus >= 0.5).astype(float))
    feddf_beta = fit_soft_head(X_public, consensus)
    hard_test = predict_head(hard_beta, X_test)
    feddf_test = predict_head(feddf_beta, X_test)

    # FedMD: repeated digest/revisit rounds on model-specific heads. Private
    # supervision and public soft consensus are balanced within each round.
    def fedmd_prediction(rounds: int) -> tuple[np.ndarray, list[dict]]:
        md_clients = [{**c, "beta": c["beta"].copy()} for c in clients]
        for _ in range(rounds):
            md_public = np.vstack([predict_head(c["beta"], c["public_features"])
                                   for c in md_clients])
            md_consensus = md_public.mean(0)
            for c in md_clients:
                features = np.vstack([c["private_features"], c["public_features"]])
                targets = np.concatenate([c["y"], md_consensus])
                weights = np.concatenate([
                    np.full(len(c["y"]), 0.5 / len(c["y"])),
                    np.full(len(md_consensus), 0.5 / len(md_consensus))])
                c["beta"] = fit_soft_head(features, targets, weights, c["beta"])
        prediction = np.mean([predict_head(c["beta"], c["test_features"])
                              for c in md_clients], axis=0)
        return prediction, md_clients

    fedmd_one_test, fedmd_one_clients = fedmd_prediction(1)
    fedmd_two_test, fedmd_two_clients = fedmd_prediction(2)
    fedmd_test, fedmd_clients = fedmd_prediction(regime.rounds)

    central_beta = fit_soft_head(X_private, y_private)
    central_test = predict_head(central_beta, X_test)
    outputs = {
        "best local": best_local,
        "unweighted ensemble": ensemble,
        "weighted ensemble": weighted_test,
        "hard transfer": hard_test,
        "FedDF": feddf_test,
        "FedMD-1r": fedmd_one_test,
        "FedMD-2r": fedmd_two_test,
        "FedMD": fedmd_test,
        "centralized reference": central_test,
    }
    best_index = int(np.argmax(local_scores))

    def live_client_prediction(client: dict) -> np.ndarray:
        return predict_head(client["beta"], client["transform"](X_test))

    def live_ensemble(model_clients: list[dict]) -> np.ndarray:
        return np.mean([live_client_prediction(client) for client in model_clients], axis=0)

    # These host diagnostics include local statevector feature evaluation for
    # VQC clients.  They exclude network and physical-QPU latency and are
    # therefore deployment-cost controls, not QPU timing claims.
    latency_ms = {
        "best local": median_runtime_ms(lambda: live_client_prediction(clients[best_index])),
        "unweighted ensemble": median_runtime_ms(lambda: live_ensemble(clients)),
        "weighted ensemble": median_runtime_ms(lambda: np.vstack(
            [live_client_prediction(client) for client in clients])),
        "hard transfer": median_runtime_ms(lambda: predict_head(hard_beta, X_test)),
        "FedDF": median_runtime_ms(lambda: predict_head(feddf_beta, X_test)),
        "FedMD-1r": median_runtime_ms(lambda: live_ensemble(fedmd_one_clients)),
        "FedMD-2r": median_runtime_ms(lambda: live_ensemble(fedmd_two_clients)),
        "FedMD": median_runtime_ms(lambda: live_ensemble(fedmd_clients)),
        "centralized reference": median_runtime_ms(
            lambda: predict_head(central_beta, X_test)),
    }
    bytes_by_method = {
        "best local": parameter_bytes(clients[best_index]),
        "unweighted ensemble": sum(parameter_bytes(c) for c in clients),
        "weighted ensemble": sum(parameter_bytes(c) for c in clients),
        "hard transfer": int(hard_beta.nbytes), "FedDF": int(feddf_beta.nbytes),
        "FedMD-1r": sum(parameter_bytes(c) for c in fedmd_one_clients),
        "FedMD-2r": sum(parameter_bytes(c) for c in fedmd_two_clients),
        "FedMD": sum(parameter_bytes(c) for c in fedmd_clients),
        "centralized reference": int(central_beta.nbytes),
    }
    # Leave-one-client-out influence is measured against the full ensemble.
    full_accuracy = metrics(ensemble, y_test)[0]
    contributions = []
    for index in range(len(clients)):
        loo = np.delete(test_predictions, index, axis=0).mean(0)
        contributions.append(full_accuracy - metrics(loo, y_test)[0])
    # Confidence-only membership inference is a deliberately weak leakage
    # indicator, reported as an attack AUC rather than a privacy guarantee.
    leakage = []
    leakage_null = []
    for c in clients:
        member = predict_head(c["beta"], c["private_features"])
        nonmember = predict_head(c["beta"], c["test_features"])
        score = np.abs(np.concatenate([member, nonmember]) - 0.5)
        labels = np.concatenate([np.ones(len(member)), np.zeros(len(nonmember))])
        leakage.append(roc_auc_score(labels, score))
        shuffled = np.random.default_rng(40_000 + seed).permutation(labels)
        leakage_null.append(roc_auc_score(shuffled, score))

    rows = []
    for method, probability in outputs.items():
        accuracy, brier, calibration = metrics(probability, y_test)
        rounds = ({"FedMD-1r": 1, "FedMD-2r": 2,
                   "FedMD": regime.rounds}.get(method, 0 if method in
                   ("best local", "centralized reference") else 1))
        models = (regime.clients if method in
                  ("unweighted ensemble", "weighted ensemble", "FedMD-1r",
                   "FedMD-2r", "FedMD")
                  else 1)
        rows.append({
            "dataset": dataset_name, "seed": seed, "method": method,
            "accuracy": accuracy, "brier": brier, "ece": calibration,
            "public_size": len(X_public), "private_size": len(X_private),
            "public_fraction": regime.public_fraction,
            "test_size": len(X_test), "clients": regime.clients,
            "noniid": regime.noniid, "noise_lambda": regime.noise_lambda,
            "shots": regime.shots, "rounds": rounds,
            "bytes": int(rounds * regime.clients * len(X_public) * 2 * 4),
            "inference_models": models, "membership_auc": float(np.mean(leakage)),
            "membership_null_auc": float(np.mean(leakage_null)),
            "inference_ms": latency_ms[method],
            "parameter_bytes": bytes_by_method[method],
            "min_client_contribution": float(np.min(contributions)),
            "mean_client_contribution": float(np.mean(contributions)),
        })
    return rows


def factorial_sensitivity() -> pd.DataFrame:
    rows = []
    for dataset_name in ("cancer", "digits", "ran-5g"):
        for noniid in ("iid", "severe"):
            for noise_lambda in (0.0, 0.45, 0.75):
                for public_fraction in (0.12, 0.24, 0.36):
                    regime = Regime(public_fraction=public_fraction, noniid=noniid,
                                    noise_lambda=noise_lambda, shots=64,
                                    clients=3, rounds=3, vqc_iters=20)
                    for seed in range(4):
                        rows.extend(run_once(dataset_name, seed, regime))
    # Client-count ablation uses one representative dataset and otherwise the
    # reference regime; architectures cycle through the three client types.
    for clients in (2, 3, 5):
        regime = Regime(clients=clients, vqc_iters=20)
        for seed in range(4):
            rows.extend(run_once("cancer", seed, regime))
    return pd.DataFrame(rows)


def mean_test_overlap() -> float:
    overlaps = []
    for name in DATASETS:
        _, y = raw_dataset(name)
        tests = []
        indices = np.arange(len(y))
        for seed in range(SEEDS):
            _, test, _, _ = train_test_split(
                indices, y, test_size=0.24, stratify=y,
                random_state=1000 + seed)
            tests.append(set(map(int, test)))
        overlaps.extend(len(a & b) / min(len(a), len(b))
                        for a, b in combinations(tests, 2))
    return float(np.mean(overlaps))


def main() -> None:
    main_rows = []
    for dataset_name in DATASETS:
        for seed in range(SEEDS):
            main_rows.extend(run_once(dataset_name, seed, MAIN))
            print(f"P2 {dataset_name} split {seed + 1}/{SEEDS}", flush=True)
    results = pd.DataFrame(main_rows)
    results.to_csv(RES / "pattern2_federation.csv", index=False)
    sensitivity = factorial_sensitivity()
    sensitivity.to_csv(RES / "pattern2_sensitivity.csv", index=False)

    summary = results.groupby("method").accuracy.mean().reindex(METHODS)
    seed_dataset = results.groupby(["method", "dataset", "seed"]).accuracy.mean().reset_index()
    errors = seed_dataset.groupby("method").accuracy.sem().reindex(METHODS)
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    ax.bar(range(len(METHODS)), summary, yerr=1.96 * errors, capsize=2,
           color=["#bbbbbb", "#66ccee", "#228833", "#ccbb44", "#4477aa",
                  "#ee7733", "#ddaa33", "#aa3377", "#999999"])
    ax.set_xticks(range(len(METHODS)), METHODS, fontsize=5.5,
                  rotation=30, ha="right")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(max(0.45, summary.min() - 0.08), 1.0)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_pattern2.pdf", bbox_inches="tight")

    weighted = results[results.method == "weighted ensemble"]
    write_macros("pattern2", {
        "PatTwoDatasets": len(DATASETS), "PatTwoSplits": SEEDS,
        "PatTwoClients": MAIN.clients, "PatTwoRounds": MAIN.rounds,
        "PatTwoNoise": f"{MAIN.noise_lambda:.2f}", "PatTwoShots": MAIN.shots,
        "PatTwoBestLocal": f"{100 * summary['best local']:.1f}\\%",
        "PatTwoEnsemble": f"{100 * summary['unweighted ensemble']:.1f}\\%",
        "PatTwoWeighted": f"{100 * summary['weighted ensemble']:.1f}\\%",
        "PatTwoHard": f"{100 * summary['hard transfer']:.1f}\\%",
        "PatTwoFedDF": f"{100 * summary['FedDF']:.1f}\\%",
        "PatTwoFedMDOne": f"{100 * summary['FedMD-1r']:.1f}\\%",
        "PatTwoFedMDTwo": f"{100 * summary['FedMD-2r']:.1f}\\%",
        "PatTwoFedMD": f"{100 * summary['FedMD']:.1f}\\%",
        "PatTwoCentral": f"{100 * summary['centralized reference']:.1f}\\%",
        "PatTwoBytes": f"{int(weighted.bytes.mean()):,}",
        "PatTwoMIA": f"{weighted.membership_auc.mean():.2f}",
        "PatTwoMIANull": f"{weighted.membership_null_auc.mean():.2f}",
        "PatTwoFedMDInferenceMs": f"{results[results.method == 'FedMD'].inference_ms.median():.3f}",
        "PatTwoFedDFInferenceMs": f"{results[results.method == 'FedDF'].inference_ms.median():.3f}",
        "PatTwoFedMDParameterBytes": f"{int(results[results.method == 'FedMD'].parameter_bytes.median()):,}",
        "PatTwoFedDFParameterBytes": f"{int(results[results.method == 'FedDF'].parameter_bytes.median()):,}",
        "PatTwoSensitivityRuns": sensitivity[["dataset", "seed", "noniid", "noise_lambda", "public_size", "clients"]].drop_duplicates().shape[0],
        "PatTwoTestOverlap": f"{100 * mean_test_overlap():.1f}\\%",
    })
    print(summary.round(4))


if __name__ == "__main__":
    main()
