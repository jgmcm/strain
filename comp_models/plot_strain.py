#!/usr/bin/env python3
"""Plot strain distributions across models, then strain maps grouped by model family."""

import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


SIZES = (4, 5)
FIGURES = ROOT / "figures"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = {4: "#2a78d6", 5: "#eb6834"}
DIVERGING = LinearSegmentedColormap.from_list(
    "compression_tension",
    ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3b3ad", "#e34948", "#8a1f1f"],
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "shear", ["#f6f3ee", "#f8cdb6", "#f19a6e", "#eb6834", "#b94716", "#6e2a0c"],
)
QUANTITIES = {
    "areal": ("Areal strain", r"$\varepsilon_A = \det F - 1$ (%)", DIVERGING),
    "shear": ("Maximum shear strain", r"$\gamma = (E_1 - E_2)/2$ (%)", SEQUENTIAL),
    "bond": ("Bond strain", r"$\varepsilon_b = \ell/\ell_0 - 1$ (%)", DIVERGING),
}

plt.rcParams.update({
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "font.size": 10,
})


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load(model, n, summary):
    directory = ROOT / "runs" / model["key"] / f"{n}x{n}"
    return {
        "model": model, "n": n, "reconstructed": summary[model["key"], n]["vacancy_reconstructed"],
        "reference": read(directory / "reference.extxyz"),
        "vacancies": read(directory / "vacancy_sites.extxyz"),
        "atoms": np.genfromtxt(directory / "strain.csv", delimiter=",", names=True),
        "bonds": np.genfromtxt(directory / "bond_strain.csv", delimiter=",", names=True),
    }


def values(item, quantity):
    if quantity == "areal":
        return 100 * item["atoms"]["areal_strain"]
    if quantity == "shear":
        return 100 * item["atoms"]["max_shear_strain"]
    return 100 * item["bonds"]["bond_strain"]


def colour_limits(items):
    """Shared limits: 99th percentile over in-domain runs with a symmetric vacancy, rounded up to 0.5 %.

    Reconstructed and out-of-domain runs saturate; panel titles give their true ranges.
    """
    typical = [item for item in items if item["model"]["domain"] == "materials" and not item["reconstructed"]]
    limits = {}
    for quantity in QUANTITIES:
        data = np.concatenate([values(item, quantity) for item in typical])
        bound = float(np.ceil(2 * np.percentile(np.abs(data), 99)) / 2)
        limits[quantity] = Normalize(0, bound) if quantity == "shear" else Normalize(-bound, bound)
    return limits


def draw_frame(ax, item):
    cell = np.asarray(item["reference"].cell)[:2, :2]
    polygon = np.array([[0, 0], cell[0], cell[0] + cell[1], cell[1]])
    for shift in ((0, 0), (1, 0), (0, 1), (1, 1)):
        for position in item["vacancies"].positions[:, :2] + np.array(shift) @ cell:
            fractional = np.linalg.solve(cell.T, position)
            if np.all(fractional >= -1e-9) and np.all(fractional <= 1 + 1e-9):
                ax.scatter(*position, marker="x", s=40, color=INK, linewidth=1.5, zorder=5)
    closed = np.vstack([polygon, polygon[0]])
    ax.plot(*closed.T, color=MUTED, linewidth=0.8, zorder=2)
    ax.set_xlim(polygon[:, 0].min() - 0.6, polygon[:, 0].max() + 0.6)
    ax.set_ylim(polygon[:, 1].min() - 0.6, polygon[:, 1].max() + 0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return polygon


def bond_segments(item):
    positions = item["reference"].positions[:, :2]
    bonds = item["bonds"]
    i, j = bonds["atom_i"].astype(int), bonds["atom_j"].astype(int)
    half = np.column_stack([bonds["x_mid_ref"], bonds["y_mid_ref"]]) - positions[i]
    # Each bond is drawn from both ends so bonds across the cell boundary appear on both sides.
    first = np.stack([positions[i], positions[i] + 2 * half], axis=1)
    second = np.stack([positions[j], positions[j] - 2 * half], axis=1)
    return np.concatenate([first, second]), np.concatenate([bonds["bond_strain"], bonds["bond_strain"]])


def draw_map(ax, item, quantity, norm):
    polygon = draw_frame(ax, item)
    clip = Polygon(polygon, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    segments, strain = bond_segments(item)
    cmap = QUANTITIES[quantity][2]
    if quantity == "bond":
        lines = LineCollection(segments, cmap=cmap, norm=norm, linewidths=3.2, capstyle="round", zorder=1)
        lines.set_array(100 * strain)
        mappable = lines
        atom_colour, atom_size = AXIS, 14
    else:
        lines = LineCollection(segments, colors=AXIS, linewidths=0.8, zorder=1)
        atom_colour, atom_size = None, 70
    ax.add_collection(lines)
    lines.set_clip_path(clip)
    positions = item["reference"].positions
    if atom_colour is None:
        mappable = ax.scatter(
            positions[:, 0], positions[:, 1], c=values(item, quantity), cmap=cmap, norm=norm,
            s=atom_size, edgecolors=MUTED, linewidths=0.4, zorder=3,
        )
    else:
        ax.scatter(positions[:, 0], positions[:, 1], s=atom_size, color=atom_colour, zorder=3)
    return mappable


def family_figure(items, family, quantity, norm):
    models = []
    for item in items:
        if item["model"]["family"] == family and item["model"] not in models:
            models.append(item["model"])
    fig, axes = plt.subplots(
        len(models), len(SIZES), figsize=(8.6, 2.75 * len(models) + 0.8),
        squeeze=False, constrained_layout=True,
    )
    lookup = {(item["model"]["key"], item["n"]): item for item in items}
    for row, model in enumerate(models):
        for column, n in enumerate(SIZES):
            ax = axes[row, column]
            mappable = draw_map(ax, lookup[model["key"], n], quantity, norm)
            data = values(lookup[model["key"], n], quantity)
            note = " · reconstructed vacancy" if lookup[model["key"], n]["reconstructed"] else ""
            span = f"{data.min():+.2f} to {data.max():+.2f} %".replace("-", "−")
            ax.set_title(f"{model['label']} · {n} × {n}\nrange {span}{note}", fontsize=10, loc="left")
    out_of_range = any(
        values(lookup[m["key"], n], quantity).max() > norm.vmax
        or values(lookup[m["key"], n], quantity).min() < norm.vmin
        for m in models for n in SIZES
    )
    extend = ("max" if quantity == "shear" else "both") if out_of_range else "neither"
    title, label, _ = QUANTITIES[quantity]
    fig.colorbar(mappable, ax=axes, label=label, shrink=0.75, extend=extend)
    fig.suptitle(f"{title} · {family}", fontsize=13, x=0.02, ha="left")
    return fig


def distribution_figure(items, models):
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.42 * len(models) + 1.6), sharey=True, constrained_layout=True)
    rows = {model["key"]: k for k, model in enumerate(models)}
    rng = np.random.default_rng(0)
    for ax, quantity in zip(axes, QUANTITIES):
        for item in items:
            data = values(item, quantity)
            offset = -0.17 if item["n"] == SIZES[0] else 0.17
            y = rows[item["model"]["key"]] + offset + rng.uniform(-0.08, 0.08, len(data))
            marker = "D" if item["reconstructed"] else "o"
            ax.scatter(data, y, s=12, marker=marker, color=SERIES[item["n"]], edgecolors="white", linewidths=0.4, zorder=3)
        if quantity != "shear":
            ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(QUANTITIES[quantity][1])
        ax.set_title(QUANTITIES[quantity][0], loc="left", fontsize=11)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        previous = None
        for k, model in enumerate(models):
            if previous is not None and model["family"] != previous:
                ax.axhline(k - 0.5, color=GRID, linewidth=0.8)
            previous = model["family"]
    axes[0].set_yticks(range(len(models)), [model["label"] + ("*" if model["domain"] != "materials" else "") for model in models])
    axes[0].set_ylim(len(models) - 0.5, -0.5)
    handles = [Line2D([], [], marker="o", linestyle="", color=SERIES[n], markersize=6, label=f"{n} × {n} cell") for n in SIZES]
    handles.append(Line2D([], [], marker="D", linestyle="", color=MUTED, markersize=5, label="reconstructed vacancy"))
    fig.legend(handles=handles, loc="outside upper right", ncols=3, frameon=False)
    fig.suptitle("Per-atom and per-bond strain across models · fixed cells at a = 2.46 Å", fontsize=13, x=0.01, ha="left")
    fig.supxlabel("One dot per atom (areal, shear) or per C–C bond (bond strain) · * out of training domain", fontsize=9, color=INK_2)
    return fig


def gallery_markdown(families):
    lines = ["::: {.panel-tabset}", ""]
    for quantity, (title, _, _) in QUANTITIES.items():
        lines += [f"### {title}", "", "::: {.panel-tabset}", ""]
        for family in families:
            lines += [f"#### {family}", "", f"![]({FIGURES.name}/{quantity}_{slug(family)}.png)", ""]
        lines += [":::", ""]
    lines.append(":::")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening a window.")
    args = parser.parse_args()
    models = json.loads((ROOT / "models.json").read_text())["models"]
    summary = {(row["model_key"], row["n"]): row for row in json.loads((ROOT / "summary.json").read_text())}
    items = [load(model, n, summary) for model in models for n in SIZES]
    FIGURES.mkdir(exist_ok=True)
    fig = distribution_figure(items, models)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(ROOT / f"strain_distributions.{extension}", dpi=200)
    limits = colour_limits(items)
    families = list(dict.fromkeys(model["family"] for model in models))
    for quantity in QUANTITIES:
        for family in families:
            figure = family_figure(items, family, quantity, limits[quantity])
            figure.savefig(FIGURES / f"{quantity}_{slug(family)}.png", dpi=150)
            plt.close(figure)
    (ROOT / "_strain_maps.md").write_text(gallery_markdown(families))
    print(f"Saved strain_distributions (PNG/SVG/PDF) and {len(QUANTITIES) * len(families)} map figures in {FIGURES.name}/.")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
