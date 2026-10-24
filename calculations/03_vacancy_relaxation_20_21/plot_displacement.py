#!/usr/bin/env python3
"""Show displacement vectors, then Gouraud-shaded magnitudes with contours."""

import argparse
from itertools import product
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list


def load(n):
    directory = ROOT / f"{n}x{n}"
    metrics = json.loads((directory / "metrics.json").read_text())
    if not metrics["converged"]:
        raise RuntimeError(f"{n}x{n} relaxation is not converged")
    reference = read(directory / "reference.extxyz")
    relaxed = read(directory / "relaxed.extxyz")
    np.testing.assert_array_equal(
        reference.arrays["original_site_index"], relaxed.arrays["original_site_index"]
    )
    data = np.genfromtxt(directory / "displacement.csv", delimiter=",", names=True)
    return {
        "n": n, "metrics": metrics, "reference": reference,
        "vacancies": read(directory / "vacancy_sites.extxyz"), "data": data,
    }


def draw_cell(ax, item, values, norm, cmap, arrows=False):
    atoms = item["reference"]
    cell = np.asarray(atoms.cell)[:2, :2]
    polygon = np.array([[0, 0], cell[0], cell[0] + cell[1], cell[1]])
    clip = Polygon(polygon, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    i, bonds = neighbor_list("iD", atoms, 1.7)
    for index, bond in zip(i, bonds):
        points = np.array([atoms.positions[index, :2], atoms.positions[index, :2] + bond[:2]])
        line, = ax.plot(*points.T, color="#d1d5db", linewidth=0.8, zorder=1)
        line.set_clip_path(clip)
    dots = ax.scatter(
        atoms.positions[:, 0], atoms.positions[:, 1], c=values,
        cmap=cmap, norm=norm, s=9, edgecolors="#374151", linewidths=0.15, zorder=3,
    )
    if arrows:
        ax.quiver(
            atoms.positions[:, 0], atoms.positions[:, 1],
            item["data"]["ux"], item["data"]["uy"],
            color="#111827", angles="xy", scale_units="xy", scale=0.2,
            width=0.0015, headwidth=3.5, zorder=4,
        )
    draw_frame(ax, item)
    return dots


def draw_frame(ax, item):
    cell = np.asarray(item["reference"].cell)[:2, :2]
    polygon = np.array([[0, 0], cell[0], cell[0] + cell[1], cell[1]])
    # Show both vacancies and equivalent corner images of the boundary vacancy.
    for shift in ((0, 0), (1, 0), (0, 1), (1, 1)):
        positions = item["vacancies"].positions[:, :2] + np.array(shift) @ cell
        for position in positions:
            fractional = np.linalg.solve(cell.T, position)
            if np.all(fractional >= -1e-9) and np.all(fractional <= 1 + 1e-9):
                ax.scatter(*position, marker="x", s=75, color="#b91c1c", linewidth=2, zorder=5)
    closed = np.vstack([polygon, polygon[0]])
    ax.plot(*closed.T, color="#111827", linestyle="--", linewidth=1.1, zorder=2)
    margin = 0.6
    ax.set_xlim(polygon[:, 0].min()-margin, polygon[:, 0].max()+margin)
    ax.set_ylim(polygon[:, 1].min()-margin, polygon[:, 1].max()+margin)
    ax.set_aspect("equal")
    ax.set_xlabel("Reference x (Å)")
    ax.set_ylabel("Reference y (Å)")
    ax.spines[["top", "right"]].set_visible(False)


def periodic_triangulation(item):
    """Triangulate the original atom values and their periodic images."""
    cell = np.asarray(item["reference"].cell)[:2, :2]
    shifts = np.array(list(product(range(-1, 2), repeat=2))) @ cell
    sites = np.concatenate([item["reference"].positions[:, :2] + shift for shift in shifts])
    magnitudes = np.tile(item["data"]["u_xy_magnitude"], len(shifts))
    # Resolve co-circular graphene sites identically in every periodic copy.
    # The perturbation selects connectivity only; plotted vertices remain exact.
    offsets = np.random.default_rng(0).normal(scale=1e-6, size=(len(item["reference"]), 2))
    tie_break_sites = sites + np.tile(offsets, (len(shifts), 1))
    connectivity = Triangulation(tie_break_sites[:, 0], tie_break_sites[:, 1]).triangles
    vertices = sites[connectivity]
    first_edge = vertices[:, 1] - vertices[:, 0]
    second_edge = vertices[:, 2] - vertices[:, 0]
    twice_area = first_edge[:, 0]*second_edge[:, 1] - first_edge[:, 1]*second_edge[:, 0]
    area_tolerance = 1e-8 * np.linalg.norm(cell, axis=1).max()**2
    connectivity = connectivity[np.abs(twice_area) > area_tolerance]
    triangulation = Triangulation(sites[:, 0], sites[:, 1], triangles=connectivity)
    vacancy_images = np.concatenate([item["vacancies"].positions[:, :2] + shift for shift in shifts])
    containing = triangulation.get_trifinder()(vacancy_images[:, 0], vacancy_images[:, 1])
    mask = np.zeros(len(triangulation.triangles), dtype=bool)
    mask[containing[containing >= 0]] = True
    triangulation.set_mask(mask)
    return triangulation, magnitudes


def save(fig, stem):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(ROOT / f"{stem}.{extension}", dpi=200, bbox_inches="tight", pad_inches=0.12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 21], help="Cell sizes to plot (default: 20 21).")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows.")
    args = parser.parse_args()
    if any(n < 1 for n in args.sizes):
        parser.error("cell sizes must be positive")
    items = [load(n) for n in args.sizes]
    maximum = max(float(item["data"]["u_xy_magnitude"].max()) for item in items)
    norm = Normalize(vmin=0, vmax=maximum)
    fig, axes = plt.subplots(1, len(items), figsize=(max(9, 6*len(items)), 5.4), constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    for ax, item in zip(axes, items):
        dots = draw_cell(ax, item, item["data"]["u_xy_magnitude"], norm, "viridis", arrows=True)
        metrics = item["metrics"]
        ax.set_title(
            f"{item['n']} × {item['n']} · {metrics['carbon_atoms']} carbon atoms\n"
            f"max |u∥| = {metrics['max_u_xy_angstrom']:.3f} Å; "
            f"fmax = {metrics['fmax_eV_per_angstrom']:.4f} eV/Å", fontsize=11,
        )
    fig.colorbar(dots, ax=axes, label="|u∥| = √(uₓ² + uᵧ²) (Å)", shrink=0.8)
    fig.suptitle("Vacancy-induced displacement · MACE-MPA-0 · fixed cells", fontsize=14)
    fig.supxlabel("Dots: surviving reference sites · arrows: (uₓ, uᵧ) × 5 · red ×: vacancy sites", fontsize=10)
    save(fig, "displacement_maps")

    fig, axes = plt.subplots(1, len(items), figsize=(max(9, 6*len(items)), 5.4), constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    line_levels = np.linspace(0, maximum, 9)[1:-1]
    for ax, item in zip(axes, items):
        triangulation, magnitudes = periodic_triangulation(item)
        filled = ax.tripcolor(triangulation, magnitudes, shading="gouraud", norm=norm, cmap="viridis")
        contours = ax.tricontour(triangulation, magnitudes, levels=line_levels,
                                 colors="#111827", linewidths=0.7, alpha=0.7)
        cell = np.asarray(item["reference"].cell)[:2, :2]
        clip = Polygon([[0, 0], cell[0], cell[0] + cell[1], cell[1]],
                       closed=True, facecolor="none", edgecolor="none")
        ax.add_patch(clip)
        filled.set_clip_path(clip)
        contours.set_clip_path(clip)
        draw_frame(ax, item)
        ax.set_title(f"{item['n']} × {item['n']} · Gouraud-shaded |u∥|", fontsize=12)
    fig.colorbar(filled, ax=axes, label="|u∥| (Å)", shrink=0.8,
                 ticks=np.linspace(0, maximum, 5), format="%.3f")
    fig.suptitle("Displacement magnitude · Gouraud shading with contour lines", fontsize=14)
    fig.supxlabel("Original atom values with periodic images · red ×: vacancy sites", fontsize=10)
    save(fig, "displacement_contours")
    print("Saved displacement vectors and Gouraud-shaded contour maps (PNG/SVG/PDF).")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
