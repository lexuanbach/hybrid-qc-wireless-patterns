"""Shared paths, plotting style, and deterministic output helpers.

Project 04 is deliberately self-contained. Quantum routines used by the
examples live in :mod:`sim_core`; no sibling project is imported.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42,
                     "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times",
                                    "Nimbus Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix"})

RES = HERE.parents[0] / "results"
# In the full project, figures and macros feed the manuscript directly; the
# standalone artifact has no submission tree, so outputs land in output/.
_PAPER = HERE.parents[0] / "submission" / "paper"
_BASE = _PAPER if _PAPER.is_dir() else HERE.parents[0] / "output"
FIG = _BASE / "figures"
GEN = _BASE / "generated"
for d in (RES, FIG, GEN):
    d.mkdir(parents=True, exist_ok=True)


def write_macros(name: str, macros: dict):
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (GEN / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(macros)} macros -> generated/{name}.tex")
