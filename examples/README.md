# Standalone reproduction artifact

From the project root, one command regenerates all CSVs, figures, LaTeX
macros, and the paper:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
make reproduce PYTHON=.venv/bin/python
```

The five study programs are also independently runnable:

- `pattern1_offload_decision.py`: 8 calibration and 24 reporting workload
  seeds; P1, utility-only, logged-IPW, oracle, LinUCB, Pareto, always-action,
  and calibration-selected fixed controls; 24 sensitivity cases.
- `wireless_ntn_study.py`: measured NetData 5G RAN load joined to declared LEO
  contact, radio, and QPU models; queue-aware, queue-blind, always-QPU, and
  always-local policies.
- `pattern2_heterogeneous_federation.py`: four main datasets, including the
  measured 5G next-window task; FedMD 1/2/3 rounds, FedDF, ensemble,
  hard-transfer, best-local, and centralized controls; public-pool, client,
  non-IID, and noise sensitivity; inference cost and membership diagnostics.
- `pattern3_subgraph_encoding.py`: exact/MILP/LNS/annealing/QAOA controls under
  patch, coupling, scope error, depth, shot, f2-noise, and 1k/5k/10k scope
  scaling variation.
- `pattern4_integrated_workflow.py`: three bundled measured Abilene weeks,
  12 randomized schedules per week, six lifecycle ablations, per-fault
  breakdowns, a signed-bias attack, verifier timing, and a deterministic matrix.
- `pattern_stats.py`: cluster bootstrap intervals whose independent units are
  workload seeds (P1/P3) or dataset means (P2), plus split-level and
  Nadeau--Bengio diagnostics.
- `check_results.py`: fast structural checks for all CSVs and generated macros.

All quantum computations are exact NumPy statevector simulations. The word
“QPU” in an action or client label denotes a declared QPU stress model; it is
not hardware execution. NetData RAN and Abilene traffic columns are measured.
Satellite radio, timing, queue, device-error, energy, price, attack, and fault
quantities are modeled.

Outputs are written to `../results/`, `../submission/paper/figures/`, and
`../submission/paper/generated/`. The artifact does not import or require any
sibling TNSE project, cloud credential, proprietary package, or network call.
