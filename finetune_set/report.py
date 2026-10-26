#!/usr/bin/env python3
"""Figures and tables for the fine-tune set review pages.

Each step writes PNG figures to figures/ and Markdown tables to steps/_*.md,
which the step pages include. Steps whose inputs do not exist yet are skipped.
"""

import argparse
import csv
import gzip
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
FIGURES = ROOT / "figures"
STEPS = ROOT / "steps"
os.environ.setdefault("MPLCONFIGDIR", str(WORK / ".mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from ase.visualize.plot import plot_atoms  # noqa: E402

INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLASS_COLOUR = {"MACE": SLOTS[0], "MatterSim": SLOTS[1], "GRACE": SLOTS[2], "PET": SLOTS[3], "DPA": SLOTS[4]}
plt.rcParams.update({"axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
                     "text.color": INK, "axes.titlecolor": INK, "font.size": 10})
ELEMENT_COLOURS = {1: "#f2f2f2", 6: "#6b6b6b", 7: "#3b5bdb", 8: "#e03131", 9: "#2f9e44"}


def log_axis(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def style(ax, grid_axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def table(header, rows, align=None):
    align = align or ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def draw_structure(ax_top, ax_side, atoms, title, values=None, cmap=None, vmax=None):
    colours = [ELEMENT_COLOURS.get(z, "#cccccc") for z in atoms.numbers]
    if values is not None:
        cm = plt.get_cmap(cmap)
        colours = [cm(min(v / vmax, 1.0)) for v in values]
    kw = dict(radii=0.32, colors=colours)
    plot_atoms(atoms, ax_top, rotation="0x,0y,0z", **kw)
    side = "-90x,0y,0z" if atoms.info.get("framework") != "nanotube" else "0x,90y,0z"
    plot_atoms(atoms, ax_side, rotation=side, show_unit_cell=0, **kw)
    ax_top.set_title(title, fontsize=8, loc="left")
    for ax in (ax_top, ax_side):
        ax.set_axis_off()


def configs_step():
    path = WORK / "initial.extxyz"
    if not path.exists():
        return
    frames = read(path, ":")
    wanted = [
        ("sheet", lambda i: i["defect_class"] == "SW" and i["curvature"] == "flat"),
        ("sheet", lambda i: i["defect_class"] == "V2"),
        ("sheet", lambda i: i["defect_class"] == "V6 hexagon"),
        ("sheet", lambda i: i["defect_class"] == "adatom" and "O" in i["defects"]),
        ("sheet", lambda i: i["defect_class"] == "H pair"),
        ("sheet", lambda i: i["curvature"] == "ripple 1D" and i["defect_class"] != "none"),
        ("sheet", lambda i: i["curvature"] == "ripple 2D"),
        ("sheet", lambda i: i["curvature"] == "bump"),
        ("sheet", lambda i: i["defect_class"] == "Frenkel pair"),
        ("nanotube", lambda i: i["defect_class"] == "SW"),
        ("nanotube", lambda i: i["defect_class"] == "adatom"),
        ("fullerene", lambda i: i["defect_class"] == "V1"),
    ]
    picks = []
    for framework, test in wanted:
        for f in frames:
            if f.info["framework"] == framework and test(f.info) and f not in picks:
                picks.append(f)
                break
    columns = 6
    rows = -(-len(picks) // columns)
    fig, axes = plt.subplots(2 * rows, columns, figsize=(2.6 * columns, 4.4 * rows),
                             gridspec_kw={"height_ratios": [2, 1.1] * rows})
    for ax in axes.ravel():
        ax.set_axis_off()
    for n, f in enumerate(picks):
        r, c = divmod(n, columns)
        i = f.info
        strain = f"ε=({i.get('strain_xx', 0):+.2f},{i.get('strain_yy', 0):+.2f},{i.get('strain_xy', 0):+.2f})" if i["framework"] == "sheet" else ""
        draw_structure(axes[2 * r, c], axes[2 * r + 1, c], f,
                       f"{i['config_id']} · {i['framework_detail']}\n{i['defects']}\n{i['curvature']} {strain}")
    fig.suptitle("Example starting configurations: top view (upper) and side view (lower)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "config_examples.png", dpi=110)
    plt.close(fig)

    summary = json.loads((ROOT / "configs_summary.json").read_text())
    natoms = [len(f) for f in frames]
    out = [f"{summary['n_configs']} configurations, {min(natoms)}–{max(natoms)} atoms (median {int(np.median(natoms))}).\n"]
    for key, title in (("framework", "Framework"), ("defect_class", "Defect class"), ("curvature", "Curvature"),
                       ("strain_mode", "Strain"), ("generator", "Assigned generator")):
        counts = Counter(f.info[key] for f in frames)
        out += [f"\n**{title}**\n", table([title, "Configurations"], sorted(counts.items(), key=lambda kv: -kv[1]), ["---", "---:"])]
    elements = Counter(e for f in frames for e in set(f.get_chemical_symbols()) if e != "C")
    out += ["\n**Configurations containing each adsorbed element**\n",
            table(["Element", "Configurations"], sorted(elements.items()), ["---", "---:"])]
    (STEPS / "_configs_tables.md").write_text("".join(out))


def sampling_step():
    logs = sorted((WORK / "sampled").glob("*/progress.log"))
    if not logs:
        return
    committee = json.loads((ROOT / "committee.json").read_text())
    labels = {m["key"]: m["label"] for m in committee["members"]}
    rows = []
    for log in logs:
        # One record per configuration: the last one, which matches the file on disk.
        latest = {}
        for line in log.open():
            record = json.loads(line)
            latest[record["config_id"]] = record
        records = list(latest.values())
        g = log.parent.name
        md = [r for r in records if r["md_frames"]]
        problems = Counter(r["status"].split(":")[0] for r in records if r["status"] != "ok")
        rows.append([labels.get(g, g), committee["generators"][g]["device"], len(records),
                     f"{np.mean([r['relax_converged'] for r in records]) * 100:.0f} %",
                     f"{np.median([r['relax_steps'] for r in records]):.0f}", len(md),
                     sum(r["frames"] for r in records), f"{sum(r['seconds'] for r in records) / 60:.0f}",
                     ", ".join(f"{k} ({v})" for k, v in problems.items()) or "none"])
    text = table(["Generator", "Device", "Configurations", "Relaxations converged", "Median FIRE steps", "MD runs",
                  "Frames written", "Compute (min)", "Stopped runs"], rows,
                 ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---"])
    summary_path = ROOT / "candidates_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        text += (f"\nAfter filtering, the pool holds **{s['frames_kept']:,} frames** "
                 f"({s['atoms_total']:,} atoms) out of {s['frames_sampled']:,} sampled.\n\n")
        text += "**Removed frames**\n\n" + table(["Stage and reason", "Frames"], sorted(s["removed"].items()), ["---", "---:"])
        text += "\n**Kept frames by stage**\n\n" + table(["Stage", "Frames"], sorted(s["kept_by_stage"].items()), ["---", "---:"])
        text += "\n**Kept frames by framework**\n\n" + table(["Framework", "Frames"], sorted(s["kept_by_framework"].items()), ["---", "---:"])
        pool = WORK / "candidates.extxyz"
        if pool.exists():
            temps = Counter(f.info.get("md_temperature_K") for f in read(pool, ":") if f.info["stage"] == "md")
            text += "\n**MD frames by temperature**\n\n" + table(["Temperature (K)", "Frames"],
                                                               [(int(t), n) for t, n in sorted(temps.items())], ["---:", "---:"])
    (STEPS / "_sampling_tables.md").write_text(text)


def read_scores():
    path = ROOT / "candidate_scores.csv.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt") as handle:
        return list(csv.DictReader(handle))


def committee_step():
    rows = read_scores()
    committee = json.loads((ROOT / "committee.json").read_text())
    members = committee["members"]
    member_rows = []
    for m in members:
        timing = ""
        path = WORK / "committee" / f"{m['key']}.npz"
        if path.exists():
            saved = np.load(path)
            # MACE members share batches; their own forward-pass time is reported.
            seconds = float(saved["model_seconds"]) if "model_seconds" in saved else float(saved["seconds"])
            timing = f"{seconds / 60:.1f}"
        member_rows.append([m["label"], m["class"], f"{m['parameters'] / 1e6:.2f}", m["level_of_theory"].replace("r2SCAN", "r²SCAN"),
                            "yes" if m["pbe_level"] else "no", m["licence"], timing])
    (STEPS / "_committee_members.md").write_text(table(
        ["Member", "Class", "Parameters (M)", "Level of theory", "In PBE score", "Licence", "Evaluation (min)"],
        member_rows, ["---", "---", "---:", "---", "---", "---", "---:"]))
    if rows is None:
        return
    summary = json.loads((ROOT / "selection_summary.json").read_text())

    # Disagreement by category.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True, constrained_layout=True)
    groups = defaultdict(list)
    for r in rows:
        stage = r["stage"] if r["stage"] != "md" else f"md {int(float(r['md_temperature_K']))} K"
        groups[(r["framework"], stage)].append(r)
    order = sorted(groups, key=lambda k: (k[0], k[1].startswith("md"), float(k[1].split()[1]) if k[1].startswith("md") else 0, k[1]))
    rng = np.random.default_rng(0)
    for ax, key, label in ((axes[0], "force_std_max", "Largest per-atom force spread, $\\max_i \\sigma_{F,i}$ (eV/Å)"),
                           (axes[1], "energy_std_per_atom", "Energy spread, $\\sigma_E$ (meV/atom)")):
        scale = 1000 if key == "energy_std_per_atom" else 1
        for y, g in enumerate(order):
            v = np.array([float(r[key]) for r in groups[g]]) * scale
            ax.scatter(v, y + rng.uniform(-0.28, 0.28, len(v)), s=3, color=SLOTS[0], alpha=0.25, linewidths=0)
            ax.plot([np.median(v)] * 2, [y - 0.38, y + 0.38], color=INK, linewidth=1.5)
        log_axis(ax)
        ax.set_xlabel(label)
        style(ax)
    axes[0].set_yticks(range(len(order)), [f"{g[0]} · {g[1]} ({len(groups[g])})" for g in order])
    axes[0].invert_yaxis()
    fig.suptitle("Committee disagreement by framework and sampling stage · PBE-level members · black bar: median",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig(FIGURES / "disagreement_by_stage.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    by_defect = defaultdict(list)
    for r in rows:
        by_defect[r["defect_class"]].append(float(r["force_std_max"]))
    order = sorted(by_defect, key=lambda k: np.median(by_defect[k]))
    for y, d in enumerate(order):
        v = np.array(by_defect[d])
        ax.scatter(v, y + rng.uniform(-0.28, 0.28, len(v)), s=3, color=SLOTS[0], alpha=0.25, linewidths=0)
        ax.plot([np.median(v)] * 2, [y - 0.38, y + 0.38], color=INK, linewidth=1.5)
    ax.set_yticks(range(len(order)), [f"{d} ({len(by_defect[d])})" for d in order])
    log_axis(ax)
    ax.set_xlabel("Largest per-atom force spread (eV/Å)")
    style(ax)
    ax.set_title("Force disagreement by defect class, all stages · black bar: median", loc="left", fontsize=12)
    fig.savefig(FIGURES / "disagreement_by_defect.png", dpi=110)
    plt.close(fig)

    loo = summary["leave_one_out_rms_force_eV_per_A"]
    ordered = sorted(members, key=lambda m: loo[m["key"]])
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for y, m in enumerate(ordered):
        ax.barh(y, loo[m["key"]], color=CLASS_COLOUR[m["class"]], height=0.7)
        ax.text(loo[m["key"]], y, f" {loo[m['key']]:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(ordered)), [m["label"] + ("" if m["pbe_level"] else " (r²SCAN)") for m in ordered])
    ax.set_xlabel("RMS force difference from the mean of the other 20 members (eV/Å)")
    style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLOUR.values()]
    ax.legend(handles, CLASS_COLOUR.keys(), frameon=False, loc="lower right")
    ax.set_title("Distance of each member from the rest of the committee", loc="left", fontsize=12)
    fig.savefig(FIGURES / "member_deviation.png", dpi=110)
    plt.close(fig)


def selection_step():
    rows = read_scores()
    curve_path = ROOT / "selection_curve.npz"
    if rows is None or not curve_path.exists():
        return
    curve = np.load(curve_path)
    d = curve["picked_distance"][1:]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    picks = np.arange(2, len(d) + 2)
    ax.plot(picks, d, color=SLOTS[0], linewidth=0.6, alpha=0.6, label="distance of each pick to the frames already chosen")
    window = 50
    smoothed = np.array([np.median(d[max(0, k - window + 1):k + 1]) for k in range(len(d))])
    ax.plot(picks, smoothed, color=INK, linewidth=2, label="running median over 50 picks")
    ax.axhline(float(curve["redundant_distance"]), color=SLOTS[1], linewidth=1.5,
               label="median distance between consecutive MD frames (20 fs)")
    ax.axvline(int(curve["n_final"]), color=INK_2, linewidth=1)
    ax.text(int(curve["n_final"]), ax.get_ylim()[1] * 0.95, f"  selected: {int(curve['n_final'])}", va="top", fontsize=9)
    ax.set_xlabel("Pick number")
    ax.set_ylabel("Descriptor distance")
    ax.set_yscale("log")
    style(ax, "y")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Greedy selection: how much new structure each pick adds", loc="left", fontsize=12)
    fig.savefig(FIGURES / "selection_curve.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), constrained_layout=True)
    for ax, key, title in zip(axes, ("defect_class", "stage", "framework"), ("Defect class", "Stage", "Framework")):
        all_counts = Counter(r[key] for r in rows)
        sel_counts = Counter(r[key] for r in rows if r["selected"] == "1")
        cats = sorted(all_counts, key=lambda c: all_counts[c])
        y = np.arange(len(cats))
        n_all, n_sel = sum(all_counts.values()), max(sum(sel_counts.values()), 1)
        ax.barh(y - 0.2, [100 * all_counts[c] / n_all for c in cats], height=0.38, color=SLOTS[0], label="candidate pool")
        ax.barh(y + 0.2, [100 * sel_counts[c] / n_sel for c in cats], height=0.38, color=SLOTS[1], label="selected")
        ax.set_yticks(y, cats)
        ax.set_xlabel("Share of frames (%)")
        ax.set_title(title, loc="left")
        style(ax)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Composition of the selected set compared with the candidate pool", x=0.01, ha="left", fontsize=12)
    fig.savefig(FIGURES / "selected_composition.png", dpi=110)
    plt.close(fig)

    selected = read(ROOT / "finetune_selected.extxyz", ":")
    top = sorted(selected, key=lambda a: -a.info["selection_score"])[:12]
    vmax = max(float(a.arrays["committee_force_std"].max()) for a in top)
    columns = 6
    rows_n = 2
    fig, axes = plt.subplots(2 * rows_n, columns, figsize=(2.6 * columns, 4.4 * rows_n),
                             gridspec_kw={"height_ratios": [2, 1.1] * rows_n})
    for ax in axes.ravel():
        ax.set_axis_off()
    for n, a in enumerate(top):
        r, c = divmod(n, columns)
        i = a.info
        stage = f"md {int(i['md_temperature_K'])} K" if i["stage"] == "md" else i["stage"]
        draw_structure(axes[2 * r, c], axes[2 * r + 1, c], a,
                       f"{i['frame_id']} · {i['generator'].split('-')[0]}\n{i['defects']} · {stage}\n"
                       f"σF max {i['committee_force_std_max']:.2f} eV/Å · σE {1000 * i['committee_energy_std_per_atom']:.0f} meV/atom",
                       values=a.arrays["committee_force_std"], cmap="Oranges", vmax=vmax)
    fig.suptitle(f"Highest-scoring selected frames · atom colour: per-atom force spread (white 0 to dark orange {vmax:.1f} eV/Å)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "top_selected.png", dpi=110)
    plt.close(fig)

    s = json.loads((ROOT / "selection_summary.json").read_text())
    text = table(["Quantity", "Value"], [
        ["Candidate frames", f"{s['candidates']:,}"],
        ["Candidate frames with H, N, O or F atoms (dropped)", f"{s['excluded_frames']:,}"],
        ["Pure-carbon frames scored", f"{s['scored']:,}"],
        ["Frames in the scored pool (top 40 % by score)", f"{s['pool_size']:,}"],
        ["Selected frames", f"{s['selected']:,}"],
        ["Natural stopping point of the greedy search", f"{s['natural_stopping_point']:,}"],
        ["Atoms in the selected frames", f"{s['selected_atoms']:,}"],
        ["Median $\\max_i \\sigma_{F,i}$, pool / selected (eV/Å)", f"{s['median_force_std_max']:.3f} / {s['selected_median_force_std_max']:.3f}"],
        ["Median $\\sigma_E$, pool / selected (meV/atom)", f"{1000 * s['median_energy_std_per_atom']:.1f} / {1000 * s['selected_median_energy_std_per_atom']:.1f}"],
    ], ["---", "---:"])
    for key, title in (("selected_by_defect_class", "Defect class"), ("selected_by_curvature", "Curvature"),
                       ("selected_by_generator", "Generator")):
        text += f"\n**Selected frames by {title.lower()}**\n\n" + table([title, "Frames"], sorted(s[key].items(), key=lambda kv: -kv[1]), ["---", "---:"])
    (STEPS / "_selection_tables.md").write_text(text)

    candidates = json.loads((ROOT / "candidates_summary.json").read_text())
    status = table(["Quantity", "Value"], [
        ["Starting configurations", "1,300"],
        ["Frames sampled by the five generators", f"{candidates['frames_sampled']:,}"],
        ["Candidate frames after filtering", f"{s['candidates']:,}"],
        ["Pure-carbon candidate frames scored", f"{s['scored']:,}"],
        ["Committee members (PBE-level members used for the score)", f"{len(s['members_pbe']) + len(s['members_r2scan'])} ({len(s['members_pbe'])})"],
        ["Selected frames", f"**{s['selected']:,}** ({s['selected_atoms']:,} atoms)"],
        ["Output file", "`finetune_set/finetune_selected.extxyz`"],
    ], ["---", "---:"])
    (STEPS / "_overview_status.md").write_text(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", default=["configs", "sampling", "committee", "selection"])
    parser.add_argument("--work", type=Path, help="Override the work directory (testing).")
    parser.add_argument("--report", type=Path, help="Directory with selection outputs and for figures and tables (testing).")
    args = parser.parse_args()
    global WORK, ROOT, FIGURES, STEPS
    if args.work:
        WORK = args.work
    if args.report:
        ROOT, FIGURES, STEPS = args.report, args.report / "figures", args.report / "steps"
    FIGURES.mkdir(exist_ok=True)
    STEPS.mkdir(exist_ok=True)
    for step in args.steps:
        {"configs": configs_step, "sampling": sampling_step, "committee": committee_step, "selection": selection_step}[step]()
        print(f"{step}: done")


if __name__ == "__main__":
    main()
