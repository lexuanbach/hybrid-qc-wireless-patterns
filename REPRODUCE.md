# Reproduce every number

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
python3 check_artifact.py                 # gate on the shipped state first

cd examples
../.venv/bin/python pattern1_offload_decision.py   # P1 two-scale offload
../.venv/bin/python wireless_ntn_study.py          # NetData 5G + LEO study
../.venv/bin/python pattern2_heterogeneous_federation.py
../.venv/bin/python pattern3_subgraph_encoding.py
../.venv/bin/python pattern4_integrated_workflow.py
../.venv/bin/python pattern_stats.py               # bootstrap intervals (CSV-only, fast)
../.venv/bin/python check_results.py               # structural integrity of outputs
```

Notes:

- Every script is deterministic (fixed seeds; stable hashing). Rerunning
  reproduces the stored `results/` CSVs; `pattern_stats.py` reproduces the
  paper's statistics macros byte-for-byte from the stored CSVs.
- Outputs land in `results/` and `output/{figures,generated}`; the shipped
  artifact contains no manuscript files, and regenerating does not add any.
- `examples/prepare_abilene_weeks.py` and
  `examples/prepare_netdata_summary.py` rebuild the bundled `data/`
  summaries from the public upstream sources recorded in the provenance
  JSONs; they are optional and require a network connection.
