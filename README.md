# Artifact: Design Patterns for Hybrid Quantum-Classical Wireless Networks

Standalone reproduction package for the T-NSE tutorial's three worked
patterns, the wireless NTN study, and the integrated Pattern-4 workflow.
It contains code, measured input data with provenance, and the stored
result CSVs behind every number, table, and figure in the paper. The
manuscript itself (LaTeX, PDF, figures) is deliberately not included.

## Layout

- `examples/` — all study programs and the shared simulator (`sim_core.py`);
  no sibling project, cloud credential, or network call is required.
- `data/` — bundled measured inputs: Abilene/Internet2 OD-traffic week
  summaries (X01/X02/X03) and the public NetData 5G RAN summary, each with a
  provenance JSON recording upstream URL and SHA-256. See `data/README.md`.
- `schemas/` — the resource-record JSON schema and two provider record
  examples used by the telemetry-contract pattern.
- `results/` — stored outputs of every study (CSV/JSON/logs). These are the
  canonical numbers; the paper's macros and figures are emitted from them.
- `requirements.txt` / `requirements-lock.txt` — Python dependencies.

## Evidence labels

All quantum computations are exact NumPy statevector simulations. "QPU" in
an action or client label denotes a declared QPU stress model, not hardware
execution. NetData RAN and Abilene traffic columns are measured; satellite
radio, timing, queue, device-error, energy, price, attack, and fault
quantities are modeled.

## One command

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python check_artifact.py       # ship-state gate
cd examples && ../.venv/bin/python pattern1_offload_decision.py   # etc.
```

See `REPRODUCE.md` for the full sequence. When run inside this artifact,
figures and LaTeX macros are written to `output/figures/` and
`output/generated/` (created on demand); result CSVs overwrite `results/`.
`examples/check_results.py` then verifies structure and macro presence.
