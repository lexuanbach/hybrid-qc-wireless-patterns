#!/usr/bin/env python3
"""Measured-RAN plus modeled-NTN timescale study for P1.

NetData supplies measured PRB, traffic, and user-count variation.  A declared
LEO geometry/radio model supplies contact-window duration, propagation,
shadowing, Rician fading, and capacity.  QPU queues and quantum quality remain
modeled.  The purpose is to test deadline-aware action admission, not to claim
field performance or quantum advantage.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from _common import FIG, RES, plt, write_macros

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "data" / "netdata_5g_summary.csv"
SEEDS = 24
EARTH_KM = 6371.0
ALTITUDE_KM = 600.0
ORBIT_RADIUS_KM = EARTH_KM + ALTITUDE_KM
MU_KM3_S2 = 398600.4418
MIN_ELEVATION_DEG = 10.0
C_M_S = 299_792_458.0
FREQUENCY_HZ = 2.0e9
BANDWIDTH_HZ = 5.0e6
EIRP_DBW = 36.0
RX_GAIN_DBI = 18.0
SYSTEM_NOISE_DBW = -128.0  # 5 MHz thermal noise plus declared receiver NF.
PAYLOAD_BITS = 2.0e6
CONTROL_DEADLINE_S = 120.0
GUARD_S = 5.0


def elevation_and_range(offset_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angular_rate = math.sqrt(MU_KM3_S2 / ORBIT_RADIUS_KM ** 3)
    central = angular_rate * offset_s
    slant = np.sqrt(ORBIT_RADIUS_KM ** 2 + EARTH_KM ** 2
                    - 2 * ORBIT_RADIUS_KM * EARTH_KM * np.cos(central))
    elevation = np.arcsin(np.clip(
        (ORBIT_RADIUS_KM * np.cos(central) - EARTH_KM) / slant, -1, 1))
    return np.degrees(elevation), slant


def contact_half_window() -> float:
    grid = np.linspace(0, 1000, 100_001)
    elevation, _ = elevation_and_range(grid)
    valid = np.flatnonzero(elevation >= MIN_ELEVATION_DEG)
    return float(grid[valid[-1]])


def simulate_context(row: pd.Series, seed: int, index: int,
                     half_window: float) -> list[dict]:
    rng = np.random.default_rng(810_000 + 1000 * seed + index)
    # Tasks span the complete usable pass instead of being conditioned on a
    # favorable elevation.  Positive offset is the descending half-pass.
    offset_s = float(rng.uniform(-half_window, half_window))
    elevation, slant_km = elevation_and_range(np.array([offset_s]))
    elevation, slant_km = float(elevation[0]), float(slant_km[0])
    remaining_s = max(0.0, half_window - offset_s)
    fspl_db = 20 * math.log10(4 * math.pi * slant_km * 1000
                              * FREQUENCY_HZ / C_M_S)
    shadow_db = float(rng.normal(0, 3.0))
    # K=7 dB Rician small-scale fading represented by its instantaneous power.
    k_linear = 10 ** (7.0 / 10)
    fading = (math.sqrt(k_linear / (k_linear + 1))
              + math.sqrt(1 / (2 * (k_linear + 1)))
              * (rng.normal() + 1j * rng.normal()))
    fading_db = 10 * math.log10(max(abs(fading) ** 2, 1e-8))
    snr_db = EIRP_DBW + RX_GAIN_DBI - fspl_db - shadow_db \
        + fading_db - SYSTEM_NOISE_DBW
    spectral_efficiency = min(6.0, math.log2(1 + 10 ** (snr_db / 10)))
    capacity_bps = BANDWIDTH_HZ * max(0.0, spectral_efficiency)
    transfer_s = (PAYLOAD_BITS / max(capacity_bps, 1.0)
                  + 2 * slant_km * 1000 / C_M_S)
    link_ok = bool(elevation >= MIN_ELEVATION_DEG and snr_db >= -5.0)

    # The queue distribution is independent of the radio fading.  Median
    # queue grows with measured RAN load to stress the joint timescale.
    load = float(row.prb_p95_pct) / 100.0
    users = float(row.users_total)
    queue_s = float(np.exp(rng.normal(math.log(18 + 55 * load), 0.75)))
    qpu_compute_s = 8.0 + 0.04 * 512
    qpu_time = queue_s + transfer_s + qpu_compute_s
    local_time = 31.0 + 12.0 * load
    admissible_s = max(0.0, min(CONTROL_DEADLINE_S, remaining_s - GUARD_S))
    local_quality = 0.082 + 0.015 * load
    qpu_quality = 0.048 + 0.010 * load
    actions = {
        "queue-aware": "qpu" if link_ok and qpu_time <= admissible_s else "local",
        "queue-blind": "qpu" if link_ok else "local",
        "always-local": "local",
        "always-qpu": "qpu",
    }
    output = []
    for policy, action in actions.items():
        duration = qpu_time if action == "qpu" else local_time
        quality = qpu_quality if action == "qpu" else local_quality
        success = bool((action == "local" or link_ok) and duration <= admissible_s)
        # Deadline failure carries a unit penalty; quality is dimensionless.
        utility = -(quality + float(not success))
        output.append({
            "seed": seed, "window": index, "timestamp": row.Timestamp,
            "policy": policy, "action": action, "success": int(success),
            "utility": utility, "elevation_deg": elevation,
            "slant_range_km": slant_km, "snr_db": snr_db,
            "capacity_mbps": capacity_bps / 1e6,
            "remaining_contact_s": remaining_s, "admissible_s": admissible_s,
            "qpu_queue_s": queue_s, "qpu_completion_s": qpu_time,
            "local_completion_s": local_time, "prb_p95_pct": row.prb_p95_pct,
            "traffic_total_kbyte": row.traffic_total_kbyte,
            "users_total": users, "ran_evidence": "measured",
            "radio_evidence": "modeled", "qpu_evidence": "modeled",
        })
    return output


def main() -> None:
    trace = pd.read_csv(TRACE)
    half_window = contact_half_window()
    rows: list[dict] = []
    for seed in range(SEEDS):
        for index, row in trace.iterrows():
            rows.extend(simulate_context(row, seed, index, half_window))
    results = pd.DataFrame(rows)
    results.to_csv(RES / "wireless_ntn.csv", index=False)
    summary = results.groupby("policy").agg(
        success=("success", "mean"), utility=("utility", "mean"),
        qpu_share=("action", lambda x: np.mean(x == "qpu")),
    ).reindex(["always-qpu", "queue-blind", "queue-aware", "always-local"])

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
    colors = ["#ee6677", "#ccbb44", "#228833", "#4477aa"]
    axes[0].bar(range(4), 100 * summary.success, color=colors)
    axes[0].set_ylabel("deadline success (%)")
    axes[0].set_ylim(0, 100)
    axes[1].bar(range(4), 100 * summary.qpu_share, color=colors)
    axes[1].set_ylabel("QPU admission (%)")
    axes[1].set_ylim(0, 100)
    labels = ["fixed\nQPU", "queue\nblind", "queue\naware", "fixed\nlocal"]
    for ax in axes:
        ax.set_xticks(range(4), labels, fontsize=5.7)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_wireless_ntn.pdf", bbox_inches="tight")

    queue = results[results.policy == "queue-aware"].qpu_queue_s
    admissible = results[results.policy == "queue-aware"].admissible_s
    aware = summary.loc["queue-aware"]
    blind = summary.loc["queue-blind"]
    write_macros("wireless_ntn", {
        "WirelessRanWindows": len(trace), "WirelessNtnSeeds": SEEDS,
        "WirelessNtnContexts": len(trace) * SEEDS,
        "WirelessContactWindow": f"{2 * half_window / 60:.1f}",
        "WirelessQueueMedian": f"{queue.median():.1f}",
        "WirelessQueueTail": f"{queue.quantile(0.95):.1f}",
        "WirelessDeadlineMedian": f"{admissible.median():.1f}",
        "WirelessAwareSuccess": f"{100 * aware.success:.1f}\\%",
        "WirelessBlindSuccess": f"{100 * blind.success:.1f}\\%",
        "WirelessAwareQpu": f"{100 * aware.qpu_share:.1f}\\%",
        "WirelessBlindQpu": f"{100 * blind.qpu_share:.1f}\\%",
        "WirelessAwareUtility": f"{aware.utility:.3f}",
        "WirelessBlindUtility": f"{blind.utility:.3f}",
    })
    print(summary.round(4))


if __name__ == "__main__":
    main()
