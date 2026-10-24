#!/usr/bin/env python3
"""Plot the unit-cell-referenced formation energy of two vacancies per cell."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Export without opening a window")
    args = parser.parse_args()
    rows = json.loads((ROOT / "formation_energies.json").read_text())
    rows = sorted((row for row in rows if row["converged"]), key=lambda row: row["n"])
    if not rows:
        raise SystemExit("No converged formation energies are available. Run relax.py first.")
    n = np.array([row["n"] for row in rows])
    energy = np.array([row["formation_energy_two_vacancies_eV"] for row in rows])
    same = np.array([row["parent_sublattices"] == "A/A" for row in rows])
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(n, energy, color="0.65", linewidth=1.2, zorder=1)
    for select, marker, color, label in (
        (~same, "o", "#2166ac", "A/B parent sublattices"),
        (same, "^", "#d95f02", "A/A parent sublattices"),
    ):
        ax.scatter(n[select], energy[select], marker=marker, color=color, s=65,
                   edgecolors="white", linewidths=0.7, label=label, zorder=3)
    ax.set_xticks(range(2, 11))
    ax.set_xlim(1.65, 10.35)
    ax.set_xlabel(r"Graphene supercell size $n$ ($n\times n$)")
    ax.set_ylabel(r"Formation energy of two vacancies (eV)")
    ax.set_title(r"$E_\mathrm{form}=E_\mathrm{defect}-(2n^2-2)\,\mu_\mathrm{C}$")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    per_vacancy = ax.secondary_yaxis("right", functions=(lambda x: x / 2, lambda x: 2 * x))
    per_vacancy.set_ylabel("Formation energy per vacancy (eV)")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(ROOT / f"formation_energy.{extension}", dpi=200, bbox_inches="tight")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
