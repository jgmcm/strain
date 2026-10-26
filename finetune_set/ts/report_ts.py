#!/usr/bin/env python3
"""Figures and tables for the transition-state review pages.

Each step writes PNG figures to ts/figures/ and Markdown tables to
ts/steps/_*.md, which the step pages include. Steps whose inputs do not exist
yet are skipped, and any include file they leave missing gets a placeholder note
so that the site still renders while the pipeline runs.
"""

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import report as base  # noqa: E402  (sets up matplotlib and the shared style)
from report import INK, INK_2, SLOTS, draw_structure, log_axis, style, table  # noqa: E402
from common import base_id  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402

WORK = ROOT / "work"
COMMITTEE = ROOT.parent / "committee.json"
FIGURES = ROOT / "figures"
STEPS = ROOT / "steps"
CLASS_ORDER = ["vacancy migration", "vacancy reconstruction", "bond rotation", "adatom migration", "recombination"]
USABLE = {"converged", "converged (loose)"}
# Include files of each step; the _text files hold prose written by hand from the results
INCLUDES = {"pairs": ["_pairs_tables.md"], "neb": ["_neb_tables.md"],
            "committee": ["_ts_committee_times.md", "_ts_committee_text.md"],
            "selection": ["_ts_selection_text.md", "_ts_selection_tables.md", "_ts_overview_status.md"]}
PENDING = "*Pending: the pipeline has not produced these results yet.*\n"


def records(path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


def pair_info():
    frames = read(WORK / "pairs.extxyz", "::2")
    return {f.info["pair_id"]: {**f.info, "carbon_only": set(f.get_chemical_symbols()) == {"C"}} for f in frames}


def pairs_step():
    path = ROOT / "pairs_summary.json"
    if not path.exists():
        return
    s = json.loads(path.read_text())
    info = pair_info()
    rows = Counter((i["framework"], i["process_class"], i["process"]) for i in info.values())
    text = (f"{s['n_pairs']:,} pairs, {s['n_atoms_min_max'][0]}–{s['n_atoms_min_max'][1]} atoms per structure "
            f"(mean {s['n_atoms_mean']:.0f}).\n\n")
    text += table(["Framework", "Process class", "Process", "Pairs"],
                  [[f, c, p, n] for (f, c, p), n in sorted(rows.items(), key=lambda kv: (kv[0][0] != "sheet", kv[0][0],
                                                                                         CLASS_ORDER.index(kv[0][1]), -kv[1]))],
                  ["---", "---", "---", "---:"])
    for key, title in (("curvature", "Curvature"), ("strain_mode", "Strain")):
        text += f"\n**{title}**\n\n" + table([title, "Pairs"], sorted(s[key].items(), key=lambda kv: -kv[1]), ["---", "---:"])
    moving = Counter(i["detail"] for i in info.values() if i["process_class"] in ("adatom migration", "recombination"))
    text += "\n**Moving atom in adatom hops and recombinations**\n\n" + table(
        ["Atom and site", "Pairs"], sorted(moving.items(), key=lambda kv: -kv[1]), ["---", "---:"])
    (STEPS / "_pairs_tables.md").write_text(text)


def band_frames(pair_ids):
    wanted, out = set(pair_ids), defaultdict(list)
    for frame in read(WORK / "bands.extxyz", ":"):
        if frame.info["pair_id"] in wanted:
            out[frame.info["pair_id"]].append(frame)
    return {k: sorted(v, key=lambda f: f.info["image"]) for k, v in out.items()}


OUTCOMES = [("usable", "Converged, with barrier"), ("no barrier", "No barrier"),
            ("split", "Split at a lower state"), ("rerun", "Rerun with more images"), ("failed", "Failed")]


def outcome(status, follow_up):
    if status in USABLE:
        return "usable"
    if status == "no barrier":
        return "no barrier"
    if status == "image below end states" and follow_up:
        return "split"
    if follow_up:
        return "rerun"
    return "failed"


def neb_step():
    ends, nebs = records(WORK / "endpoints.jsonl"), records(WORK / "neb.jsonl")
    if not nebs:
        return
    info = pair_info()
    by_process = defaultdict(Counter)
    for r in ends:
        key = (info[r["pair_id"]]["framework"], info[r["pair_id"]]["process"])
        by_process[key]["pairs"] += 1
        by_process[key][r["status"]] += 1
    steps, usable_pairs = defaultdict(list), defaultdict(set)
    for r in nebs:
        key = (info[base_id(r["pair_id"])]["framework"], r["process"])
        by_process[key]["bands"] += 1
        by_process[key][outcome(r["status"], r["follow_up"])] += 1
        if r["status"] in USABLE:
            steps[key].append(r["steps"])
            usable_pairs[key].add(base_id(r["pair_id"]))
    rows, total = [], Counter()
    for key in sorted(by_process, key=lambda k: (k[0] != "sheet", k[0], k[1])):
        c = by_process[key]
        c["pairs with a transition state"] = len(usable_pairs[key])
        total.update(c)
        rows.append([key[0], key[1], c["pairs"], c["ok"], c["bands"], *[c[o] for o, _ in OUTCOMES],
                     c["pairs with a transition state"], f"{np.median(steps[key]):.0f}" if steps[key] else ""])
    all_steps = [r["steps"] for r in nebs if r["status"] in USABLE]
    rows.append(["**all**", "", *[f"**{total[k]}**" for k in ("pairs", "ok", "bands")],
                 *[f"**{total[o]}**" for o, _ in OUTCOMES], f"**{total['pairs with a transition state']}**",
                 f"{np.median(all_steps):.0f}" if all_steps else ""])
    text = table(["Framework", "Process", "Pairs", "End states usable", "Bands run", *[label for _, label in OUTCOMES],
                  "Pairs with a transition state", "Median steps"],
                 rows, ["---", "---", *["---:"] * (len(OUTCOMES) + 5)])
    end_counts = Counter(r["status"] for r in ends)
    text += "\n**Pairs removed after the end-state relaxation**\n\n" + table(
        ["Reason", "Pairs"], [[r, n] for r, n in sorted(end_counts.items()) if r != "ok"], ["---", "---:"])
    rounds = Counter((r["generation"], r["status"]) for r in nebs)
    text += "\n**Band outcomes by round**\n\n" + table(
        ["Round", "Outcome", "Bands"], [[g + 1, s, n] for (g, s), n in sorted(rounds.items())], ["---:", "---", "---:"])
    (STEPS / "_neb_tables.md").write_text(text)

    # The figures show the pure-carbon bands, the ones that go on to the committee.
    usable = [r for r in nebs if r["status"] in USABLE and info[base_id(r["pair_id"])]["carbon_only"]]
    if not usable:
        return
    # Generator barriers by process.
    groups = defaultdict(list)
    for r in usable:
        groups[(info[base_id(r["pair_id"])]["process_class"], r["process"])].append(r["barrier_forward"])
    order = sorted(groups, key=lambda k: (CLASS_ORDER.index(k[0]), np.median(groups[k])))
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(order) + 1.6), constrained_layout=True)
    rng = np.random.default_rng(0)
    for y, key in enumerate(order):
        v = np.array(groups[key])
        ax.scatter(np.maximum(v, 0.01), y + rng.uniform(-0.28, 0.28, len(v)), s=6,
                   color=SLOTS[CLASS_ORDER.index(key[0])], alpha=0.5, linewidths=0)
        ax.plot([np.median(v)] * 2, [y - 0.38, y + 0.38], color=INK, linewidth=1.5)
    ax.set_yticks(range(len(order)), [f"{k[1]} ({len(groups[k])})" for k in order])
    ax.invert_yaxis()
    log_axis(ax)
    ax.set_xlabel("Forward barrier from MACE-OMAT-0 small, $E_\\mathrm{TS} - E_\\mathrm{initial}$ (eV)")
    style(ax)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=SLOTS[n]) for n in range(len(CLASS_ORDER))]
    ax.legend(handles, CLASS_ORDER, frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Generator barriers by process, pure-carbon bands, all frameworks · black bar: median", loc="left",
                 fontsize=12)
    fig.savefig(FIGURES / "generator_barriers.png", dpi=110)
    plt.close(fig)

    # Example bands: one per process class and framework.
    picks, seen = [], set()
    for r in sorted(usable, key=lambda r: r["pair_id"]):
        i = info[base_id(r["pair_id"])]
        key = (i["framework"], i["process_class"]) if i["framework"] == "sheet" else (i["framework"],)
        if key not in seen:
            seen.add(key)
            picks.append(r)
    picks = picks[:9]
    bands = band_frames([r["pair_id"] for r in picks])
    fig, axes = plt.subplots(len(picks), 4, figsize=(13, 2.9 * len(picks)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    for row, r in zip(axes, picks):
        images = bands[r["pair_id"]]
        top = r["climbing_image"]
        event = set(np.atleast_1d(images[0].info["event_atoms"]).astype(int))
        colours = [SLOTS[1] if n in event else base.ELEMENT_COLOURS.get(z, "#cccccc") for n, z in enumerate(images[0].numbers)]
        rotation = "0x,0y,0z" if images[0].info["framework"] != "nanotube" else "0x,90y,0z"
        for ax, image, title in zip(row[:3], (images[0], images[top], images[-1]), ("initial", "climbing image", "final")):
            base.plot_atoms(image, ax, rotation=rotation, radii=0.32, colors=colours)
            ax.set_axis_off()
            ax.set_title(title, fontsize=8, loc="left")
        e = np.array(r["energies"]) - r["energies"][0]
        path = np.concatenate([[0], np.cumsum([np.linalg.norm(images[m + 1].positions - images[m].positions)
                                                for m in range(len(images) - 1)])])
        ax = row[3]
        ax.plot(path, e, color=SLOTS[0], marker="o", markersize=4, linewidth=1.5)
        ax.plot(path[top], e[top], marker="o", markersize=8, color=SLOTS[1])
        ax.set_xlabel("Path length (Å)", fontsize=8)
        ax.set_ylabel("$E - E_\\mathrm{initial}$ (eV)", fontsize=8)
        style(ax, "y")
        i = info[base_id(r["pair_id"])]
        ax.set_title(f"{r['pair_id']} · {i['framework_detail']} · {r['process']}\n{i['context']} · {i['detail']} · "
                     f"{i['curvature']} · barrier {r['barrier_forward']:.2f} eV", fontsize=8, loc="left")
    fig.suptitle("Example bands from MACE-OMAT-0 small · orange: atoms moved by the process", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "example_bands.png", dpi=100)
    plt.close(fig)


def read_scores():
    path = ROOT / "candidate_scores.csv.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt") as handle:
        return list(csv.DictReader(handle))


def committee_step():
    rows = read_scores()
    committee = json.loads(COMMITTEE.read_text())
    members = committee["members"]
    timing_rows = []
    for m in members:
        path = WORK / "committee" / f"{m['key']}.npz"
        if path.exists():
            saved = np.load(path)
            seconds = float(saved["model_seconds"]) if "model_seconds" in saved else float(saved["seconds"])
            timing_rows.append([m["label"], f"{seconds / 60:.1f}"])
    if timing_rows:
        (STEPS / "_ts_committee_times.md").write_text(table(["Member", "Evaluation (min)"], timing_rows, ["---", "---:"]))
    if rows is None:
        return
    ts_rows = [r for r in rows if r["barrier_std"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True, constrained_layout=True)
    groups = defaultdict(list)
    for r in ts_rows:
        groups[(r["process_class"], r["process"])].append(r)
    order = sorted(groups, key=lambda k: (CLASS_ORDER.index(k[0]), k[1]))
    rng = np.random.default_rng(0)
    for ax, key, label in ((axes[0], "force_std_max", "Largest per-atom force spread, $\\max_i \\sigma_{F,i}$ (eV/Å)"),
                           (axes[1], "barrier_std", "Barrier spread of the band, $\\sigma_B$ (eV)")):
        # The barrier spread is one value per band, shown at the climbing image.
        stages = ("ts",) if key == "barrier_std" else ("ts", "ts_neighbour", "ts_displaced")
        for y, g in enumerate(order):
            for stage, colour in zip(("ts", "ts_neighbour", "ts_displaced"), (SLOTS[1], SLOTS[0], SLOTS[2])):
                v = np.array([float(r[key]) for r in groups[g] if r["stage"] == stage and stage in stages])
                if len(v):
                    ax.scatter(v, y + rng.uniform(-0.28, 0.28, len(v)), s=4, color=colour, alpha=0.35, linewidths=0)
            v = np.array([float(r[key]) for r in groups[g] if r["stage"] in stages])
            ax.plot([np.median(v)] * 2, [y - 0.38, y + 0.38], color=INK, linewidth=1.5)
        log_axis(ax)
        ax.set_xlabel(label)
        style(ax)
    axes[0].set_yticks(range(len(order)), [f"{g[1]} ({len(groups[g])})" for g in order])
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c) for c in (SLOTS[1], SLOTS[0], SLOTS[2])]
    axes[1].legend(handles, ["climbing image", "neighbouring image", "displaced climbing image"], frameon=False,
                   fontsize=8, loc="lower right")
    fig.suptitle("Committee disagreement in the transition-state region · PBE-level members · black bar: median",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig(FIGURES / "ts_disagreement_by_process.png", dpi=110)
    plt.close(fig)

    # Each member's barrier relative to the committee median, per climbing image.
    with gzip.open(ROOT / "barriers.csv.gz", "rt") as handle:
        barrier_rows = list(csv.DictReader(handle))
    values = np.array([[float(r[m["key"]]) for m in members] for r in barrier_rows])
    median = np.median(values[:, [k for k, m in enumerate(members) if m["pbe_level"]]], axis=1)
    deviation = values - median[:, None]
    order = np.argsort(np.median(deviation, axis=0))
    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    for y, k in enumerate(order):
        m = members[k]
        v = deviation[:, k]
        lo, q1, q2, q3, hi = np.percentile(v, [10, 25, 50, 75, 90])
        colour = base.CLASS_COLOUR[m["class"]]
        ax.plot([lo, hi], [y, y], color=colour, linewidth=1)
        ax.add_patch(plt.Rectangle((q1, y - 0.3), q3 - q1, 0.6, color=colour, alpha=0.8, linewidth=0))
        ax.plot([q2, q2], [y - 0.34, y + 0.34], color=INK, linewidth=1.5)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(range(len(order)), [members[k]["label"] + ("" if members[k]["pbe_level"] else " (r²SCAN)") for k in order])
    ax.set_xlabel("Member barrier minus the median barrier of the 18 PBE-level members (eV)")
    style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in base.CLASS_COLOUR.values()]
    ax.legend(handles, base.CLASS_COLOUR.keys(), frameon=False, loc="lower right")
    ax.set_title(f"Member barriers relative to the committee, {len(barrier_rows):,} climbing images\n"
                 "box: 25–75 %, line: 10–90 %, black bar: median", loc="left", fontsize=11)
    fig.savefig(FIGURES / "member_barrier_deviation.png", dpi=110)
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
    ax.plot(picks, d, color=SLOTS[0], linewidth=0.6, alpha=0.6,
            label="distance of each climbing image to those already chosen")
    window = 50
    smoothed = np.array([np.median(d[max(0, k - window + 1):k + 1]) for k in range(len(d))])
    ax.plot(picks, smoothed, color=INK, linewidth=2, label="running median over 50 picks")
    ax.axhline(float(curve["redundant_distance"]), color=SLOTS[1], linewidth=1.5,
               label="median distance between a climbing image and its neighbouring images")
    ax.axvline(int(curve["n_final"]), color=INK_2, linewidth=1)
    ax.text(int(curve["n_final"]), ax.get_ylim()[1] * 0.95, f"  selected: {int(curve['n_final'])} bands", va="top",
            fontsize=9)
    ax.set_xlabel("Pick number (climbing images)")
    ax.set_ylabel("Descriptor distance")
    ax.set_yscale("log")
    style(ax, "y")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Greedy selection: how much new structure each pick adds", loc="left", fontsize=12)
    fig.savefig(FIGURES / "ts_selection_curve.png", dpi=110)
    plt.close(fig)

    ts_rows = [r for r in rows if r["barrier_std"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), constrained_layout=True)
    for ax, key, title in zip(axes, ("process", "stage", "framework"), ("Process", "Frame", "Framework")):
        all_counts = Counter(r[key] for r in ts_rows)
        sel_counts = Counter(r[key] for r in ts_rows if r["selected"] == "1")
        cats = sorted(all_counts, key=lambda c: all_counts[c])
        y = np.arange(len(cats))
        n_all, n_sel = sum(all_counts.values()), max(sum(sel_counts.values()), 1)
        ax.barh(y - 0.2, [100 * all_counts[c] / n_all for c in cats], height=0.38, color=SLOTS[0], label="transition-state frames")
        ax.barh(y + 0.2, [100 * sel_counts[c] / n_sel for c in cats], height=0.38, color=SLOTS[1], label="selected")
        ax.set_yticks(y, cats)
        ax.set_xlabel("Share of frames (%)")
        ax.set_title(title, loc="left")
        style(ax)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Composition of the selected set compared with all transition-state frames", x=0.01, ha="left", fontsize=12)
    fig.savefig(FIGURES / "ts_selected_composition.png", dpi=110)
    plt.close(fig)

    selected = read(ROOT / "finetune_ts_selected.extxyz", ":")
    top = sorted([a for a in selected if a.info["stage"] == "ts"], key=lambda a: -a.info["selection_score"])[:12]
    vmax = max(float(a.arrays["committee_force_std"].max()) for a in top)
    columns, rows_n = 6, 2
    fig, axes = plt.subplots(2 * rows_n, columns, figsize=(2.6 * columns, 4.4 * rows_n),
                             gridspec_kw={"height_ratios": [2, 1.1] * rows_n})
    for ax in axes.ravel():
        ax.set_axis_off()
    for n, a in enumerate(top):
        r, c = divmod(n, columns)
        i = a.info
        draw_structure(axes[2 * r, c], axes[2 * r + 1, c], a,
                       f"{i['frame_id']} · {i['framework_detail']}\n{i['process']}\n"
                       f"σF max {i['committee_force_std_max']:.2f} eV/Å · σB {i['committee_barrier_std']:.2f} eV",
                       values=a.arrays["committee_force_std"], cmap="Oranges", vmax=vmax)
    fig.suptitle(f"Highest-scoring selected climbing images · atom colour: per-atom force spread (white 0 to dark orange {vmax:.1f} eV/Å)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "ts_top_selected.png", dpi=110)
    plt.close(fig)

    s = json.loads((ROOT / "selection_summary.json").read_text())
    text = table(["Quantity", "Value"], [
        ["Bands with both end states and a transition-state region", f"{s['bands']:,}"],
        ["Transition-state frames (climbing, neighbouring and displaced images)", f"{s['transition_state_frames']:,}"],
        ["Climbing images in the scored pool (top 50 % by score)", f"{s['pool_size']:,}"],
        ["Natural stopping point of the greedy search (bands)", f"{s['natural_stopping_point']:,}"],
        ["Selected bands", f"{s['selected_bands']:,}"],
        ["Selected frames (climbing image and one other image per band)", f"{s['selected']:,}"],
        ["Atoms in the selected frames", f"{s['selected_atoms']:,}"],
        ["Median $\\max_i \\sigma_{F,i}$ of climbing images, all / selected bands (eV/Å)",
         f"{s['median_force_std_max_climbing']:.3f} / {s['selected_median_force_std_max_climbing']:.3f}"],
        ["Median $\\sigma_B$, all / selected bands (eV)",
         f"{s['median_barrier_std']:.3f} / {s['selected_median_barrier_std']:.3f}"],
    ], ["---", "---:"])
    for key, title in (("selected_by_process", "Process"), ("selected_by_stage", "Frame"),
                       ("selected_by_framework", "Framework"), ("selected_by_curvature", "Curvature")):
        text += f"\n**Selected frames by {title.lower()}**\n\n" + table(
            [title, "Frames"], sorted(s[key].items(), key=lambda kv: -kv[1]), ["---", "---:"])
    (STEPS / "_ts_selection_tables.md").write_text(text)

    pairs = json.loads((ROOT / "pairs_summary.json").read_text())
    candidates = json.loads((ROOT / "candidates_summary.json").read_text())
    status = table(["Quantity", "Value"], [
        ["Pairs of end states built", f"{pairs['n_pairs']:,}"],
        ["Bands run, including reruns and split bands", f"{candidates['bands']:,}"],
        ["Pure-carbon bands with a converged climbing image and a barrier", f"{candidates['usable_bands']:,}"],
        ["Such bands with H, N, O or F atoms (dropped)",
         f"{sum(candidates.get('usable_bands_with_other_elements', {}).values()):,}"],
        ["Transition-state frames scored by the committee", f"{s['transition_state_frames']:,}"],
        ["Selected frames", f"**{s['selected']:,}** ({s['selected_atoms']:,} atoms)"],
        ["Output file", "`finetune_set/ts/finetune_ts_selected.extxyz`"],
    ], ["---", "---:"])
    (STEPS / "_ts_overview_status.md").write_text(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", default=["pairs", "neb", "committee", "selection"])
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
        {"pairs": pairs_step, "neb": neb_step, "committee": committee_step, "selection": selection_step}[step]()
        for name in INCLUDES[step]:
            if not (STEPS / name).exists():
                (STEPS / name).write_text(PENDING)
        print(f"{step}: done")


if __name__ == "__main__":
    main()
