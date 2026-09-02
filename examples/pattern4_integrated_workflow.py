#!/usr/bin/env python3
"""Trace-driven P1/P4/P5/P6/fallback and stale-loop mechanism study.

Only Abilene traffic columns are measured. Provider telemetry and injected
faults are modeled and tagged accordingly. HMAC is a test-fixture integrity
mechanism, not a production key-management claim.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator, FormatChecker

from _common import FIG, GEN, RES, plt, write_macros

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "data" / "abilene_three_weeks_summary.csv"
SCHEMA = json.loads((ROOT / "schemas" / "resource-record-v1.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())
TEST_KEY = b"tnse-tutorial-test-fixture-key"
FAULTS = ("none", "stale-record", "replay", "corruption", "infeasible-proposal",
          "planner-bypass", "stale-calibration", "link-outage")
RANDOM_FAULTS = FAULTS + ("signed-bias",)
CLOCK_SKEW_S = 2.0
AUTHORIZATION = {
    "planner-service": {("capacity.allocate", "ran-slice:demo")},
}
CONTROLLERS = ("unguarded", "typed", "authenticated", "verified",
               "guarded", "guarded-stable")


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def canonical(record: dict) -> bytes:
    payload = deepcopy(record); payload.pop("signature", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def sign(record: dict) -> dict:
    record = deepcopy(record)
    record["signature"] = {
        "algorithm": "HMAC-SHA256-test-fixture", "key_id": "fixture-key",
        "value": hmac.new(TEST_KEY, canonical(record), hashlib.sha256).hexdigest(),
    }
    return record


def metric(name: str, value: float, unit: str, observed_at: str,
           evidence: str, relative_uncertainty: float = 0.15) -> dict:
    radius = abs(value) * relative_uncertainty
    return {"name": name, "value": float(value), "unit": unit,
            "observed_at": observed_at, "evidence": evidence,
            "uncertainty": {"kind": "interval", "lower": float(value - radius),
                            "upper": float(value + radius), "coverage": 0.95}}


def normalize_provider_a(raw: dict, timestamp: str, nonce: str) -> dict:
    """Map seconds/fractions/USD fields from provider A into schema v1."""
    valid = (utc(timestamp) + timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": "1.0", "provider": "provider-a", "backend": raw["device"],
        "generated_at": timestamp, "valid_until": valid, "nonce": nonce,
        "capabilities": {"qubits": raw["qubits"], "max_shots": raw["shot_limit"]},
        "metrics": [metric("queue_delay", raw["queue_seconds"], "s", timestamp, "modeled"),
                    metric("two_qubit_error", raw["cx_error_fraction"], "1", timestamp, "claimed"),
                    metric("price_per_shot", raw["usd_per_shot"], "USD/shot", timestamp, "claimed", 0)],
        "provenance": {"producer": "provider-a-adapter", "method": "field-map-a-v1",
                       "source_id": raw["source_id"]},
    }


def normalize_provider_b(raw: dict, timestamp: str, nonce: str) -> dict:
    """Map milliseconds/percent/micro-USD fields from provider B into v1."""
    valid = (utc(timestamp) + timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": "1.0", "provider": "provider-b", "backend": raw["backend_name"],
        "generated_at": timestamp, "valid_until": valid, "nonce": nonce,
        "capabilities": {"qubits": raw["num_qubits"], "max_shots": raw["max_samples"]},
        "metrics": [metric("queue_delay", raw["wait_ms"] / 1000, "s", timestamp, "modeled"),
                    metric("two_qubit_error", raw["cx_error_percent"] / 100, "1", timestamp, "claimed"),
                    metric("price_per_shot", raw["micro_usd_per_sample"] / 1e6,
                           "USD/shot", timestamp, "claimed", 0)],
        "provenance": {"producer": "provider-b-adapter", "method": "field-map-b-v1",
                       "source_id": raw["snapshot"]},
    }


def telemetry(index: int, timestamp: str, rng: np.random.Generator) -> dict:
    queue = float(np.exp(rng.normal(np.log(0.45 + 0.25 * (index % 11 == 0)), 0.55)))
    error = float(np.clip(rng.normal(0.014, 0.006), 0.002, 0.08))
    nonce = f"telemetry-{index:05d}"
    if index % 2 == 0:
        raw = {"device": "qpu-a1", "queue_seconds": queue,
               "cx_error_fraction": error, "usd_per_shot": 1.0e-6,
               "qubits": 127, "shot_limit": 100000, "source_id": f"a-{index}"}
        return sign(normalize_provider_a(raw, timestamp, nonce))
    raw = {"backend_name": "quantum-b7", "wait_ms": 1000 * queue,
           "cx_error_percent": 100 * error, "micro_usd_per_sample": 1.4,
           "num_qubits": 84, "max_samples": 50000, "snapshot": f"b-{index}"}
    return sign(normalize_provider_b(raw, timestamp, nonce))


def values(record: dict) -> dict:
    return {item["name"]: item["value"] for item in record["metrics"]}


def validate_schema(record: dict) -> tuple[bool, str]:
    errors = list(VALIDATOR.iter_errors(record))
    if errors:
        return False, "schema"
    return True, "accepted"


def validate(record: dict, event_time: datetime, seen_nonces: set[str],
             distributional: bool = False) -> tuple[bool, str]:
    structural, reason = validate_schema(record)
    if not structural:
        return structural, reason
    expected = hmac.new(TEST_KEY, canonical(record), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, record["signature"]["value"]):
        return False, "signature"
    if record["nonce"] in seen_nonces:
        return False, "replay"
    if event_time > utc(record["valid_until"]) + timedelta(seconds=CLOCK_SKEW_S):
        return False, "expired"
    for item in record["metrics"]:
        age = event_time - utc(item["observed_at"])
        if age > timedelta(minutes=10, seconds=CLOCK_SKEW_S) or \
                age < timedelta(seconds=-CLOCK_SKEW_S):
            return False, "stale-metric"
        if item["uncertainty"]["lower"] > item["value"] or item["uncertainty"]["upper"] < item["value"]:
            return False, "uncertainty"
    if values(record)["two_qubit_error"] > 0.08:
        return False, "calibration"
    if distributional and values(record)["queue_delay"] < 0.05:
        return False, "distribution-shift"
    seen_nonces.add(record["nonce"])
    return True, "accepted"


def inject(record: dict, fault: str, prior_nonce: str | None) -> tuple[dict, bool]:
    record = deepcopy(record); link_available = True
    if fault == "stale-record":
        old = (utc(record["generated_at"]) - timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
        record["generated_at"] = old; record["valid_until"] = old
        record = sign(record)
    elif fault == "replay" and prior_nonce:
        record["nonce"] = prior_nonce; record = sign(record)
    elif fault == "corruption":
        record["metrics"][0]["value"] *= 0.01  # modified after signature
    elif fault == "stale-calibration":
        old = (utc(record["generated_at"]) - timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
        for item in record["metrics"]:
            if item["name"] == "two_qubit_error": item["observed_at"] = old
        record = sign(record)
    elif fault == "link-outage":
        link_available = False
    elif fault == "signed-bias":
        # A dishonest signer reports an internally consistent but implausibly
        # small queue.  Authentication alone cannot detect this fault.
        for item in record["metrics"]:
            if item["name"] == "queue_delay":
                item["value"] *= 0.02
                item["uncertainty"]["lower"] = 0.8 * item["value"]
                item["uncertainty"]["upper"] = 1.2 * item["value"]
        record = sign(record)
    return record, link_available


def candidate(current: float, history: list[float], action: str, fault: str) -> dict:
    growth = 0.0 if len(history) < 2 else np.clip((history[-1] - history[-2]) / history[-2], -0.1, 0.15)
    allocation = current * (1.10 + max(0, growth))
    if fault == "infeasible-proposal": allocation = 0.72 * current
    return {"allocation": float(allocation), "action": action,
            "principal": ("unknown-planner" if fault == "planner-bypass"
                          else "planner-service"),
            "scope": "capacity.allocate", "resource": "ran-slice:demo",
            "bypass": fault == "planner-bypass"}


def verify_proposal(proposal: dict, current: float) -> tuple[bool, str]:
    permission = (proposal["scope"], proposal["resource"])
    if proposal["bypass"] or permission not in AUTHORIZATION.get(
            proposal["principal"], set()):
        return False, "authorization"
    if not np.isfinite(proposal["allocation"]) or proposal["allocation"] < current: return False, "capacity"
    if proposal["allocation"] > 2.5 * current: return False, "resource-limit"
    return True, "accepted"


def select_action(record: dict, link_available: bool) -> tuple[str, float]:
    telemetry_values = values(record)
    remote_score = -(telemetry_values["queue_delay"] + 6 * telemetry_values["two_qubit_error"])
    local_score = -0.78
    return (("remote", remote_score - local_score) if link_available
            else ("local", -np.inf)) if remote_score > local_score and link_available else ("local", local_score - remote_score)


def run_controller(trace: pd.DataFrame, controller: str, hysteresis: float = 0.08,
                   min_hold: int = 2, telemetry_delay: int = 0,
                   fault_schedule: list[str] | None = None,
                   replay_id: int = 0) -> pd.DataFrame:
    rows, seen, history = [], set(), []
    previous_action, last_switch, last_accepted_nonce = "local", -10, None
    audit_previous = "0" * 64
    for index, point in trace.iterrows():
        event_time = utc(point.timestamp + "Z")
        source_index = max(0, index - telemetry_delay)
        source_time = trace.iloc[source_index].timestamp + "Z"
        rng = np.random.default_rng(90_000 + index + 10_000 * replay_id)
        record = telemetry(source_index, source_time, rng)
        actual_values = values(record)
        actual_queue = actual_values["queue_delay"]
        actual_error = actual_values["two_qubit_error"]
        fault = (fault_schedule[index] if fault_schedule is not None
                 else FAULTS[index % len(FAULTS)])
        record, link_available = inject(record, fault, last_accepted_nonce)
        check_start = time.perf_counter_ns()
        if controller == "unguarded":
            structurally_valid, reason = True, "unchecked"
        elif controller == "typed":
            structurally_valid, reason = validate_schema(record)
        else:
            structurally_valid, reason = validate(
                record, event_time, seen,
                distributional=controller in ("guarded", "guarded-stable"))
        telemetry_check_us = (time.perf_counter_ns() - check_start) / 1000
        if structurally_valid:
            action, margin = select_action(record, link_available)
        else:
            action, margin = "fallback", np.inf
        if controller == "guarded-stable" and action != "fallback" and action != previous_action:
            if margin < hysteresis or index - last_switch < min_hold:
                action = previous_action
            else:
                last_switch = index
        if action != previous_action and action != "fallback":
            switched = 1; previous_action = action
        else:
            switched = 0
        proposal = candidate(float(point.total_load), history, action, fault)
        verify_start = time.perf_counter_ns()
        if controller in ("verified", "guarded", "guarded-stable"):
            verified, verify_reason = verify_proposal(proposal, float(point.total_load))
        else:
            verified, verify_reason = True, "unchecked"
        proposal_check_us = (time.perf_counter_ns() - verify_start) / 1000
        fallback = not structurally_valid or not verified or (action == "fallback")
        allocation = 1.25 * float(point.total_load) if fallback else proposal["allocation"]
        permission = (proposal["scope"], proposal["resource"])
        authorized = permission in AUTHORIZATION.get(proposal["principal"], set())
        unauthorized_commit = bool(verified and not authorized)
        underallocation_commit = bool(verified and
                                      allocation + 1e-12 < float(point.next_total_load))
        unsafe = bool(unauthorized_commit or underallocation_commit)
        actual_remote_score = -(actual_queue + 6 * actual_error)
        bad_venue = bool(fault == "signed-bias" and action == "remote"
                         and actual_remote_score <= -0.78)
        recovery = 300 if unsafe else 0
        output = {
            "week": str(point.week), "window": index, "replay_id": replay_id,
            "controller": controller, "fault": fault,
            "action": action, "telemetry_valid": structurally_valid,
            "telemetry_reason": reason, "proposal_valid": verified,
            "proposal_reason": verify_reason, "fallback": int(fallback),
            "unsafe_commit": int(unsafe and verified), "sla_violation": int(unsafe),
            "unauthorized_commit": int(unauthorized_commit),
            "underallocation_commit": int(underallocation_commit),
            "bad_venue": int(bad_venue),
            "recovery_s": recovery, "switch": switched,
            "traffic_evidence": "measured", "telemetry_evidence": "modeled",
            "telemetry_delay": telemetry_delay, "hysteresis": hysteresis,
            "min_hold": min_hold,
            "telemetry_check_us": telemetry_check_us,
            "proposal_check_us": proposal_check_us,
            "verification_us": telemetry_check_us + proposal_check_us,
            "audit_prev_hash": audit_previous,
        }
        encoded = json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
        output["audit_hash"] = hashlib.sha256(
            audit_previous.encode() + encoded).hexdigest()
        audit_previous = output["audit_hash"]
        rows.append(output)
        history.append(float(point.total_load))
        if structurally_valid:
            last_accepted_nonce = record["nonce"]
    return pd.DataFrame(rows)


def balanced_schedule(length: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    schedule = list(RANDOM_FAULTS) * int(np.ceil(length / len(RANDOM_FAULTS)))
    rng.shuffle(schedule)
    return schedule[:length]


def verify_audit_chain(frame: pd.DataFrame) -> None:
    previous = "0" * 64
    for row in frame.to_dict("records"):
        claimed = row.pop("audit_hash")
        if row["audit_prev_hash"] != previous:
            raise RuntimeError("broken audit predecessor")
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        computed = hashlib.sha256(previous.encode() + encoded).hexdigest()
        if computed != claimed:
            raise RuntimeError("broken audit hash")
        previous = claimed


def main() -> None:
    trace = pd.read_csv(TRACE)
    # Static fixtures prove that both normalized provider examples satisfy the
    # same structural contract. Dynamic records additionally pass signatures.
    for name in ("provider-a-record.json", "provider-b-record.json"):
        fixture = json.loads((ROOT / "schemas" / name).read_text())
        VALIDATOR.validate(fixture)

    integrated_parts = []
    for controller in CONTROLLERS:
        for week, week_trace in trace.groupby("week", sort=False):
            part = run_controller(week_trace.reset_index(drop=True), controller)
            verify_audit_chain(part)
            integrated_parts.append(part)
    integrated = pd.concat(integrated_parts, ignore_index=True)
    integrated.to_csv(RES / "pattern4_integrated.csv", index=False)
    randomized_parts = []
    for replay_id in range(12):
        for controller in CONTROLLERS:
            for week_index, (week, week_trace) in enumerate(
                    trace.groupby("week", sort=False)):
                schedule = balanced_schedule(len(week_trace),
                                             700_000 + 100 * replay_id + week_index)
                part = run_controller(week_trace.reset_index(drop=True), controller,
                                      fault_schedule=schedule, replay_id=replay_id + 1)
                verify_audit_chain(part)
                randomized_parts.append(part)
    randomized = pd.concat(randomized_parts, ignore_index=True)
    randomized.to_csv(RES / "pattern4_randomized.csv", index=False)
    breakdown = randomized.groupby(["controller", "fault"]).agg(
        windows=("window", "size"), unsafe=("unsafe_commit", "sum"),
        unauthorized=("unauthorized_commit", "sum"),
        underallocation=("underallocation_commit", "sum"),
        bad_venue=("bad_venue", "sum"), fallback=("fallback", "mean"),
    ).reset_index()
    breakdown.to_csv(RES / "pattern4_fault_breakdown.csv", index=False)
    indexed = breakdown.set_index(["controller", "fault"])
    latex_rows = []
    for fault in RANDOM_FAULTS:
        raw = indexed.loc[("unguarded", fault)]
        full = indexed.loc[("guarded-stable", fault)]
        label = fault.replace("-", " ")
        latex_rows.append(
            f"{label} & {int(raw.unsafe)} & {int(full.unsafe)} & "
            f"{int(full.unauthorized)} & {int(full.underallocation)} & "
            f"{int(full.bad_venue)} \\\\"
        )
    (GEN / "pattern4_breakdown.tex").write_text("\n".join(latex_rows) + "\n")
    stress = []
    for delay in (0, 1, 3, 6):
        for hysteresis in (0.0, 0.08, 0.16):
            for hold in (0, 2):
                part = run_controller(trace[trace.week == "X01"].reset_index(drop=True),
                                      "guarded-stable", hysteresis, hold, delay)
                part["configuration"] = f"d{delay}-h{hysteresis}-r{hold}"
                stress.append(part)
    stress = pd.concat(stress, ignore_index=True)
    stress.to_csv(RES / "pattern4_stability.csv", index=False)

    summary = integrated.groupby("controller").agg(
        unsafe=("unsafe_commit", "mean"), sla=("sla_violation", "mean"),
        fallback=("fallback", "mean"), recovery=("recovery_s", "mean"),
        switches=("switch", "sum")).reindex(CONTROLLERS)
    random_summary = randomized.groupby("controller").agg(
        unsafe=("unsafe_commit", "mean"), bad_venue=("bad_venue", "mean"),
        fallback=("fallback", "mean"), recovery=("recovery_s", "mean"),
        switches=("switch", "sum")).reindex(CONTROLLERS)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
    colors = ["#ee6677", "#eeaa55", "#ccbb44", "#66ccee", "#4477aa", "#228833"]
    axes[0].bar(range(6), 100 * random_summary.unsafe, color=colors)
    axes[0].set_xticks(range(6), ["select", "+P6", "+P5", "+P4", "+dist.", "full"],
                      fontsize=5.2, rotation=25)
    axes[0].set_ylabel("unsafe commits (%)")
    axes[0].grid(axis="y", lw=0.3, alpha=0.5)
    axes[1].bar(range(6), 100 * random_summary.bad_venue, color=colors)
    axes[1].set_xticks(range(6), ["select", "+P6", "+P5", "+P4", "+dist.", "full"],
                      fontsize=5.2, rotation=25)
    axes[1].set_ylabel("bad venue choices (%)")
    axes[1].grid(axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_integrated.pdf", bbox_inches="tight")

    write_macros("pattern4", {
        "IntegratedWindows": len(trace), "IntegratedWeeks": trace.week.nunique(),
        "IntegratedFaults": len(FAULTS), "IntegratedRandomFaults": len(RANDOM_FAULTS),
        "IntegratedReplayReplications": randomized.replay_id.nunique(),
        "IntegratedUnsafeRaw": f"{100 * summary.loc['unguarded', 'unsafe']:.1f}\\%",
        "IntegratedUnsafeGuard": f"{100 * summary.loc['guarded', 'unsafe']:.1f}\\%",
        "IntegratedSlaGuard": f"{100 * summary.loc['guarded', 'sla']:.1f}\\%",
        "IntegratedFallbackGuard": f"{100 * summary.loc['guarded', 'fallback']:.1f}\\%",
        "IntegratedRecoveryRaw": f"{summary.loc['unguarded', 'recovery']:.0f}",
        "IntegratedRecoveryGuard": f"{summary.loc['guarded', 'recovery']:.0f}",
        "IntegratedSwitchRaw": int(summary.loc['unguarded', 'switches']),
        "IntegratedSwitchStable": int(summary.loc['guarded-stable', 'switches']),
        "IntegratedStressCases": stress.configuration.nunique(),
        "IntegratedVerifyMedianUs": f"{randomized[randomized.controller == 'guarded'].verification_us.median():.1f}",
        "IntegratedVerifyTailUs": f"{randomized[randomized.controller == 'guarded'].verification_us.quantile(.95):.1f}",
        "IntegratedBenignFallback": f"{100 * randomized[(randomized.controller == 'guarded') & (randomized.fault == 'none')].fallback.mean():.1f}\\%",
        "IntegratedBadVenueVerified": f"{100 * random_summary.loc['verified', 'bad_venue']:.1f}\\%",
        "IntegratedBadVenueGuarded": f"{100 * random_summary.loc['guarded', 'bad_venue']:.1f}\\%",
        "IntegratedRandomUnsafeRaw": f"{100 * random_summary.loc['unguarded', 'unsafe']:.1f}\\%",
        "IntegratedRandomUnsafeTyped": f"{100 * random_summary.loc['typed', 'unsafe']:.1f}\\%",
        "IntegratedRandomUnsafeAuthenticated": f"{100 * random_summary.loc['authenticated', 'unsafe']:.1f}\\%",
        "IntegratedRandomUnsafeVerified": f"{100 * random_summary.loc['verified', 'unsafe']:.1f}\\%",
        "IntegratedRandomUnsafeGuarded": f"{100 * random_summary.loc['guarded', 'unsafe']:.1f}\\%",
        "IntegratedRandomUnsafeFull": f"{100 * random_summary.loc['guarded-stable', 'unsafe']:.1f}\\%",
    })
    print("deterministic")
    print(summary.round(4))
    print("randomized")
    print(random_summary.round(4))


if __name__ == "__main__":
    main()
