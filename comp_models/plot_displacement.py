#!/usr/bin/env python3
"""Gouraud-shaded displacement magnitudes with contours for every model, as in Figure 1.

Each model gets two versions of the calculation 02 contour figure: one on a
colour scale shared by all models and one on its own scale. The script also
draws all models on the shared scale for each cell size, and writes the model
browser included by notes.qmd.
"""

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


SIZES = (4, 5)
FIGURES = ROOT / "figures" / "displacement"
COMMON_STEP = 0.01  # Angstrom; the shared maximum is rounded up to this step


def load(model, n, summary):
    directory = ROOT / "runs" / model["key"] / f"{n}x{n}"
    reference = read(directory / "reference.extxyz")
    relaxed = read(directory / "relaxed.extxyz")
    np.testing.assert_array_equal(
        reference.arrays["original_site_index"], relaxed.arrays["original_site_index"]
    )
    return {
        "model": model, "n": n, "summary": summary[model["key"], n],
        "reference": reference, "vacancies": read(directory / "vacancy_sites.extxyz"),
        "data": np.genfromtxt(directory / "displacement.csv", delimiter=",", names=True),
    }


def draw_frame(ax, item, labels=True):
    cell = np.asarray(item["reference"].cell)[:2, :2]
    polygon = np.array([[0, 0], cell[0], cell[0] + cell[1], cell[1]])
    # Show both vacancies and equivalent corner images of the boundary vacancy.
    for shift in ((0, 0), (1, 0), (0, 1), (1, 1)):
        positions = item["vacancies"].positions[:, :2] + np.array(shift) @ cell
        for position in positions:
            fractional = np.linalg.solve(cell.T, position)
            if np.all(fractional >= -1e-9) and np.all(fractional <= 1 + 1e-9):
                ax.scatter(*position, marker="x", s=75 if labels else 30, color="#b91c1c",
                           linewidth=2 if labels else 1.4, zorder=5)
    closed = np.vstack([polygon, polygon[0]])
    ax.plot(*closed.T, color="#111827", linestyle="--", linewidth=1.1, zorder=2)
    margin = 0.6
    ax.set_xlim(polygon[:, 0].min()-margin, polygon[:, 0].max()+margin)
    ax.set_ylim(polygon[:, 1].min()-margin, polygon[:, 1].max()+margin)
    ax.set_aspect("equal")
    if labels:
        ax.set_xlabel("Reference x (Å)")
        ax.set_ylabel("Reference y (Å)")
        ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.set_axis_off()


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


def draw_contours(ax, item, norm, labels=True):
    triangulation, magnitudes = periodic_triangulation(item)
    filled = ax.tripcolor(triangulation, magnitudes, shading="gouraud", norm=norm, cmap="viridis")
    line_levels = np.linspace(0, norm.vmax, 9)[1:-1]
    contours = ax.tricontour(triangulation, magnitudes, levels=line_levels,
                             colors="#111827", linewidths=0.7 if labels else 0.5, alpha=0.7)
    cell = np.asarray(item["reference"].cell)[:2, :2]
    clip = Polygon([[0, 0], cell[0], cell[0] + cell[1], cell[1]],
                   closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    filled.set_clip_path(clip)
    contours.set_clip_path(clip)
    draw_frame(ax, item, labels)
    return filled


def colour_bar(fig, mappable, axes, norm, items, shrink):
    saturated = any(item["data"]["u_xy_magnitude"].max() > norm.vmax + 1e-12 for item in items)
    fig.colorbar(mappable, ax=axes, label="|u∥| (Å)", shrink=shrink, extend="max" if saturated else "neither",
                 ticks=np.linspace(0, norm.vmax, 5), format="%.3f")


def model_figure(items, norm, scale_note):
    model = items[0]["model"]
    fig, axes = plt.subplots(1, len(items), figsize=(6*len(items), 5.4), constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    for ax, item in zip(axes, items):
        filled = draw_contours(ax, item, norm)
        note = " · reconstructed vacancy" if item["summary"]["vacancy_reconstructed"] else ""
        ax.set_title(
            f"{item['n']} × {item['n']} · Gouraud-shaded |u∥|\n"
            f"max |u∥| = {item['summary']['max_u_xy_angstrom']:.3f} Å{note}", fontsize=12,
        )
    colour_bar(fig, filled, axes, norm, items, 0.8)
    fig.suptitle(f"Displacement magnitude · {model['label']} · {scale_note}", fontsize=14)
    fig.supxlabel("Original atom values with periodic images · red ×: vacancy sites", fontsize=10)
    return fig


def overview_figure(items, n, norm):
    selected = [item for item in items if item["n"] == n]
    columns = 4
    rows = -(-len(selected) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(13, 2.75 * rows + 0.8), constrained_layout=True, squeeze=False)
    for ax in axes.ravel()[len(selected):]:
        ax.set_axis_off()
    for ax, item in zip(axes.ravel(), selected):
        filled = draw_contours(ax, item, norm, labels=False)
        marks = ("*" if item["model"]["domain"] != "materials" else "") + (
            " · reconstructed" if item["summary"]["vacancy_reconstructed"] else "")
        ax.set_title(f"{item['model']['label']}{marks}\nmax |u∥| = {item['summary']['max_u_xy_angstrom']:.3f} Å",
                     fontsize=9.5, loc="left")
    colour_bar(fig, filled, axes, norm, selected, 0.6)
    fig.suptitle(f"Displacement magnitude · all models · {n} × {n} · shared colour scale", fontsize=14)
    fig.supxlabel("Gouraud shading of atom values with contour lines · red ×: vacancy sites · * out of training domain",
                  fontsize=10)
    return fig


def browser_markdown(models, summary):
    entries = []
    for model in models:
        cells = [summary[model["key"], n] for n in SIZES]
        entries.append({
            "key": model["key"], "label": model["label"], "family": model["family"],
            "theory": model["level_of_theory"].replace("r2SCAN", "r²SCAN").replace("wB97M", "ωB97M"),
            "dtype": cells[0]["dtype"], "outOfDomain": model["domain"] != "materials",
            "cells": [{"n": c["n"], "maxU": round(c["max_u_xy_angstrom"], 3), "steps": c["steps"],
                       "reconstructed": c["vacancy_reconstructed"]} for c in cells],
        })
    groups = []
    for family in dict.fromkeys(entry["family"] for entry in entries):
        options = "".join(
            f'<option value="{e["key"]}">{e["label"]}</option>' for e in entries if e["family"] == family
        )
        groups.append(f'<optgroup label="{family}">{options}</optgroup>')
    options = "".join(groups)
    return f"""```{{=html}}
<div id="displacement-browser" class="displacement-browser" tabindex="-1">
<div class="d-flex flex-wrap gap-2 align-items-center mb-2">
<button type="button" class="btn btn-sm btn-outline-secondary" data-step="-1" aria-label="Previous model">← Previous</button>
<select class="form-select form-select-sm w-auto" aria-label="Model">{options}</select>
<button type="button" class="btn btn-sm btn-outline-secondary" data-step="1" aria-label="Next model">Next →</button>
<div class="btn-group btn-group-sm" role="group" aria-label="Colour scale">
<input type="radio" class="btn-check" name="displacement-scale" id="displacement-scale-common" value="common" checked>
<label class="btn btn-outline-secondary" for="displacement-scale-common">Shared scale</label>
<input type="radio" class="btn-check" name="displacement-scale" id="displacement-scale-own" value="own">
<label class="btn btn-outline-secondary" for="displacement-scale-own">Model's own scale</label>
</div>
</div>
<p class="small text-body-secondary mb-2" data-role="caption" aria-live="polite"></p>
<img data-role="image" class="img-fluid" alt="">
</div>
<script>
(() => {{
  const models = {json.dumps(entries, ensure_ascii=False)};
  const root = document.getElementById("displacement-browser");
  const select = root.querySelector("select");
  const image = root.querySelector('[data-role="image"]');
  const caption = root.querySelector('[data-role="caption"]');
  let index = models.findIndex((m) => m.key === "mace-mpa-0-medium");
  const scale = () => root.querySelector('input[name="displacement-scale"]:checked').value;
  const describe = (c) => `${{c.n}} × ${{c.n}}: max |u∥| ${{c.maxU.toFixed(3)}} Å, ${{c.steps}} steps` +
    (c.reconstructed ? ", reconstructed vacancy" : "");
  function render() {{
    const m = models[index];
    select.value = m.key;
    image.src = `{FIGURES.relative_to(ROOT).as_posix()}/${{scale()}}_${{m.key}}.png`;
    image.alt = `Gouraud-shaded displacement magnitude with contour lines for ${{m.label}}, 4 × 4 and 5 × 5 cells`;
    caption.textContent = `${{index + 1}} of ${{models.length}} · ${{m.label}}` +
      (m.outOfDomain ? " (outside training domain)" : "") +
      ` · ${{m.theory}} · ${{m.dtype}} · ` + m.cells.map(describe).join(" · ");
  }}
  const step = (delta) => {{ index = (index + delta + models.length) % models.length; render(); }};
  root.querySelectorAll("button[data-step]").forEach((b) => b.addEventListener("click", () => step(Number(b.dataset.step))));
  select.addEventListener("change", () => {{ index = models.findIndex((m) => m.key === select.value); render(); }});
  root.querySelectorAll('input[name="displacement-scale"]').forEach((r) => r.addEventListener("change", render));
  root.addEventListener("keydown", (event) => {{
    if (event.target.tagName === "SELECT") return;
    if (event.key === "ArrowLeft") {{ step(-1); event.preventDefault(); }}
    if (event.key === "ArrowRight") {{ step(1); event.preventDefault(); }}
  }});
  render();
}})();
</script>
```
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows.")
    args = parser.parse_args()
    models = json.loads((ROOT / "models.json").read_text())["models"]
    summary = {(row["model_key"], row["n"]): row for row in json.loads((ROOT / "summary.json").read_text())}
    items = [load(model, n, summary) for model in models for n in SIZES]
    typical = [item for item in items
               if item["model"]["domain"] == "materials" and not item["summary"]["vacancy_reconstructed"]]
    common_max = np.ceil(max(item["data"]["u_xy_magnitude"].max() for item in typical) / COMMON_STEP) * COMMON_STEP
    common = Normalize(vmin=0, vmax=common_max)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for model in models:
        pair = [item for item in items if item["model"]["key"] == model["key"]]
        own = Normalize(vmin=0, vmax=max(float(item["data"]["u_xy_magnitude"].max()) for item in pair))
        for name, norm, note in (("common", common, f"shared scale, 0–{common_max:.2f} Å"),
                                 ("own", own, "model's own scale")):
            fig = model_figure(pair, norm, note)
            fig.savefig(FIGURES / f"{name}_{model['key']}.png", dpi=120)
            plt.close(fig)
    for n in SIZES:
        fig = overview_figure(items, n, common)
        # No SVG: Gouraud meshes for 19 models give SVG files of 15-25 MB.
        for extension in ("png", "pdf"):
            fig.savefig(ROOT / f"displacement_contours_all_{n}x{n}.{extension}", dpi=150)
    (ROOT / "_displacement_browser.md").write_text(browser_markdown(models, summary))
    print(f"Shared scale 0-{common_max:.2f} Angstrom; saved {2 * len(models)} model figures in "
          f"{FIGURES.relative_to(ROOT)}/ and the all-model overviews (PNG/PDF).")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
