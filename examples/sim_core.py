"""Small, standalone simulation primitives used by the tutorial examples.

The quantum routines below are exact NumPy statevector calculations. They do
not call quantum hardware. Network delay, QPU error, energy, and price are
explicit stochastic models in the calling experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize


def bit_table(n: int) -> np.ndarray:
    states = np.arange(1 << n, dtype=np.uint32)
    return ((states[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int8)


def normalized_cost(costs: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(costs)), float(np.max(costs))
    return np.zeros_like(costs, dtype=float) if hi <= lo else (costs - lo) / (hi - lo)


def _apply_mixer(state: np.ndarray, beta: float, n: int) -> np.ndarray:
    c, s = math.cos(beta), -1j * math.sin(beta)
    for q in range(n):
        stride = 1 << q
        view = state.reshape(-1, stride << 1)
        a, b = view[:, :stride].copy(), view[:, stride:].copy()
        view[:, :stride] = c * a + s * b
        view[:, stride:] = s * a + c * b
    return state


def qaoa_probabilities(costs: np.ndarray, n: int, parameters: np.ndarray) -> np.ndarray:
    """Exact statevector probabilities for X-mixer QAOA at depth p."""
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 1 or len(parameters) == 0 or len(parameters) % 2:
        raise ValueError("parameters must contain p gammas followed by p betas")
    depth = len(parameters) // 2
    norm = normalized_cost(costs)
    state = np.ones(1 << n, dtype=np.complex128) / math.sqrt(1 << n)
    for layer in range(depth):
        state *= np.exp(-1j * parameters[layer] * norm)
        state = _apply_mixer(state, parameters[depth + layer], n)
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def tune_qaoa(costs: np.ndarray, n: int, rng: np.random.Generator,
              depth: int = 1, shots: int = 256, max_evals: int = 89,
              noise_lambda: float = 0.0) -> dict:
    """Tune QAOA and sample under an optional global depolarizing channel.

    ``noise_lambda`` mixes the ideal final distribution with the uniform
    distribution.  This circuit-level f2 model is deliberately hardware
    agnostic; it does not model compilation or correlated device noise.
    """
    start = time.perf_counter()
    norm = normalized_cost(costs)
    evaluations = 0

    def probabilities(theta: np.ndarray) -> np.ndarray:
        ideal = qaoa_probabilities(costs, n, theta)
        return (1 - noise_lambda) * ideal + noise_lambda / len(ideal)

    def objective(theta: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return float(probabilities(theta) @ norm)

    if depth == 1:
        starts = [np.array([g, b])
                  for g in np.linspace(0, 2 * np.pi, 7, endpoint=False)
                  for b in np.linspace(0, np.pi, 7, endpoint=False)]
    else:
        # Deterministic low-discrepancy-like starts keep p=2 within the same
        # total objective-evaluation budget as p=1.
        starts = [np.array([g, 0.5 * g, b, 0.5 * b])
                  for g in np.linspace(0, 2 * np.pi, 4, endpoint=False)
                  for b in np.linspace(0, np.pi, 4, endpoint=False)]
    values = [objective(theta) for theta in starts]
    theta0 = starts[int(np.argmin(values))]
    remaining = max(1, max_evals - evaluations)
    result = minimize(objective, theta0, method="Nelder-Mead",
                      options={"maxfev": remaining, "xatol": 1e-3, "fatol": 1e-5})
    theta = result.x if result.fun < min(values) else theta0
    probs = probabilities(theta)
    samples = rng.choice(len(probs), size=shots, p=probs)
    best = int(samples[np.argmin(costs[samples])])
    return {
        "solution": best,
        "cost": float(costs[best]),
        "evals": evaluations,
        "shots": shots,
        "depth": depth,
        "noise_lambda": noise_lambda,
        "success_prob": float(probs[costs <= costs.min() + 1e-10].sum()),
        "runtime_s": time.perf_counter() - start,
    }


def _apply_ry_batch(states: np.ndarray, q: int, angles: np.ndarray) -> None:
    stride = 1 << q
    view = states.reshape(states.shape[0], -1, stride << 1)
    a, b = view[:, :, :stride].copy(), view[:, :, stride:].copy()
    c, s = np.cos(angles / 2)[:, None, None], np.sin(angles / 2)[:, None, None]
    view[:, :, :stride] = c * a - s * b
    view[:, :, stride:] = s * a + c * b


def _apply_cz_ring(states: np.ndarray, n: int) -> None:
    if n < 2:
        return
    indices = np.arange(states.shape[1])
    bits = (indices[:, None] >> np.arange(n)) & 1
    pairs = [(q, (q + 1) % n) for q in range(n)] if n > 2 else [(0, 1)]
    sign = np.ones(states.shape[1])
    for a, b in pairs:
        sign *= np.where((bits[:, a] == 1) & (bits[:, b] == 1), -1.0, 1.0)
    states *= sign[None, :]


def vqc_forward(X: np.ndarray, theta: np.ndarray, n: int = 4,
                layers: int = 2, reupload: bool = False) -> np.ndarray:
    """Return all per-qubit Z expectations from an exact statevector."""
    X = np.asarray(X, dtype=float)
    if X.shape[1] < n:
        X = np.pad(X, ((0, 0), (0, n - X.shape[1])))
    states = np.zeros((len(X), 1 << n), dtype=np.complex128)
    states[:, 0] = 1.0
    for q in range(n):
        _apply_ry_batch(states, q, X[:, q])
    weights = np.asarray(theta).reshape(layers, n)
    for layer in range(layers):
        if reupload and layer:
            for q in range(n):
                _apply_ry_batch(states, q, 0.5 * X[:, q])
        for q in range(n):
            _apply_ry_batch(states, q, np.full(len(X), weights[layer, q]))
        _apply_cz_ring(states, n)
    probs = np.abs(states) ** 2
    indices = np.arange(1 << n)
    features = np.empty((len(X), n))
    for q in range(n):
        features[:, q] = probs @ (1.0 - 2.0 * ((indices >> q) & 1))
    return features


@dataclass
class ChannelInstance:
    n: int
    fields: np.ndarray
    weights: np.ndarray
    costs: np.ndarray


def channel_instance(n: int, seed: int, coupling: float = 1.0) -> ChannelInstance:
    """Binary channel assignment with unary and pairwise interference costs."""
    rng = np.random.default_rng(seed)
    mask = np.triu(rng.random((n, n)) < min(0.55, 3.2 / max(n - 1, 1)), 1)
    weights = np.triu(rng.uniform(0.4, 1.8, (n, n)), 1) * mask * coupling
    fields = rng.uniform(-0.35, 0.35, n)
    bits = bit_table(n)
    same = bits[:, :, None] == bits[:, None, :]
    costs = (same * weights[None, :, :]).sum(axis=(1, 2)) + bits @ fields
    return ChannelInstance(n=n, fields=fields, weights=weights, costs=costs.astype(float))


def costs_from_terms(fields: np.ndarray, weights: np.ndarray) -> np.ndarray:
    bits = bit_table(len(fields))
    same = bits[:, :, None] == bits[:, None, :]
    return (same * weights[None, :, :]).sum(axis=(1, 2)) + bits @ fields


def local_search(costs: np.ndarray, n: int, rng: np.random.Generator,
                 starts: int = 8) -> tuple[int, int]:
    evaluations, best = 0, None
    for _ in range(starts):
        state = int(rng.integers(1 << n))
        improved = True
        while improved:
            improved = False
            candidates = np.array([state ^ (1 << q) for q in range(n)], dtype=int)
            evaluations += n
            nxt = int(candidates[np.argmin(costs[candidates])])
            if costs[nxt] + 1e-12 < costs[state]:
                state, improved = nxt, True
        if best is None or costs[state] < costs[best]:
            best = state
    return int(best), evaluations


def anneal(costs: np.ndarray, n: int, rng: np.random.Generator,
           budget: int, initial: int | None = None) -> tuple[int, int]:
    state = int(rng.integers(1 << n) if initial is None else initial)
    best = state
    for step in range(max(1, budget)):
        candidate = state ^ (1 << int(rng.integers(n)))
        temperature = max(1e-3, 1.0 - step / max(1, budget))
        delta = float(costs[candidate] - costs[state])
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            state = candidate
        if costs[state] < costs[best]:
            best = state
    return best, budget


def channel_milp(fields: np.ndarray, weights: np.ndarray) -> dict:
    """Solve the quadratic binary channel objective by exact MILP linearization."""
    start = time.perf_counter()
    n = len(fields)
    edges = [(i, j, float(weights[i, j])) for i in range(n)
             for j in range(i + 1, n) if weights[i, j] != 0]
    # w(1-x_i-x_j+2y_ij), with y_ij=x_i*x_j.
    c = np.concatenate([fields.copy(), np.array([2 * w for _, _, w in edges])])
    for i, j, w in edges:
        c[i] -= w
        c[j] -= w
    rows, lower, upper = [], [], []
    for k, (i, j, _) in enumerate(edges):
        y = n + k
        row = np.zeros(n + len(edges)); row[y] = 1; row[i] = -1
        rows.append(row); lower.append(-np.inf); upper.append(0)
        row = np.zeros(n + len(edges)); row[y] = 1; row[j] = -1
        rows.append(row); lower.append(-np.inf); upper.append(0)
        row = np.zeros(n + len(edges)); row[y] = 1; row[i] = -1; row[j] = -1
        rows.append(row); lower.append(-1); upper.append(np.inf)
    constraints = LinearConstraint(np.asarray(rows), np.asarray(lower), np.asarray(upper))
    result = milp(c=c, integrality=np.ones(len(c)),
                  bounds=Bounds(np.zeros(len(c)), np.ones(len(c))),
                  constraints=constraints, options={"time_limit": 30})
    bits = np.rint(result.x[:n]).astype(int)
    state = int(np.dot(bits, 1 << np.arange(n)))
    return {"solution": state, "success": bool(result.success),
            "runtime_s": time.perf_counter() - start}
