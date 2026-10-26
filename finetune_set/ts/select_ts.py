#!/usr/bin/env python3
"""Score committee disagreement in the transition-state region and select frames for fine-tuning.

Disagreement is measured over the PBE-level members, as for the first set:

- force spread: per atom, sigma_i = sqrt(mean_m |F_mi - mean_m F_mi|^2); per frame, its
  maximum and mean over atoms;
- barrier spread: for each member, the climbing image's energy relative to the mean
  of the band's two end-state energies, E_m - (E_m(initial) + E_m(final)) / 2; per
  band, the standard deviation over members (eV). The end states have the same atoms,
  so no elemental reference energies are needed. Every frame of a band carries the
  band's barrier spread. A displaced copy's own energy spread is not used, because it
  mostly measures how stiff each member is against the random displacement.

The score adds the two spreads, each divided by its median over the climbing
images. Bands are chosen through their climbing images, with the
diversity-weighted greedy search of the first set (select_frames.py) over the
top `--pool-fraction` of climbing images by score. The number of bands is where
the running median of the pick distance falls to the median distance between a
climbing image and its neighbouring images, bounded so that the frame count
stays in the requested range. Each chosen band contributes its climbing image
and the neighbouring or displaced image with the largest force spread.
"""

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from select_frames import force_spread, greedy_selection, leave_one_out_rms, load_predictions, reduce_frames  # noqa: E402

WORK = ROOT / "work"
TS_STAGES = ("ts", "ts_neighbour", "ts_displaced")
NEAR_STAGES = ("ts_neighbour", "ts_displaced")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-frames", type=int, default=500)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--pool-fraction", type=float, default=0.5)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--report", type=Path, default=ROOT)
    args = parser.parse_args()
    work, report = args.work, args.report

    committee = json.loads((ROOT.parent / "committee.json").read_text())
    members = committee["members"]
    frames = read(work / "candidates.extxyz", ":")
    natoms = np.array([len(f) for f in frames])
    offsets = np.concatenate([[0], np.cumsum(natoms)])
    energies, forces = load_predictions(members, natoms, work)
    pbe = np.array([m["pbe_level"] for m in members])
    sigma_atom, mean_forces, sigma_max, sigma_mean = force_spread(forces[pbe], offsets)
    _, _, sigma_max_all, _ = force_spread(forces, offsets)

    # Energies relative to the mean of each band's end states, per member.
    stage = np.array([f.info["stage"] for f in frames])
    band = np.array([f.info["pair_id"] for f in frames])
    ends = {}
    for k in np.flatnonzero((stage == "initial") | (stage == "final")):
        ends.setdefault(band[k], {})[stage[k]] = k
    complete = {b for b, e in ends.items() if len(e) == 2}
    region = np.array([s in TS_STAGES and b in complete for s, b in zip(stage, band)])
    relative = np.full(energies.shape, np.nan)
    barrier = np.full(energies.shape, np.nan)  # forward barrier, E_m - E_m(initial)
    for k in np.flatnonzero(region):
        e = ends[band[k]]
        relative[:, k] = energies[:, k] - (energies[:, e["initial"]] + energies[:, e["final"]]) / 2
        barrier[:, k] = energies[:, k] - energies[:, e["initial"]]
    energy_std = np.where(region, np.nanstd(np.where(region, relative, 0.0)[pbe], axis=0), np.nan)
    energy_std_all = np.where(region, np.nanstd(np.where(region, relative, 0.0), axis=0), np.nan)
    # The barrier spread is the energy spread of the band's climbing image.
    top_of = {band[k]: k for k in np.flatnonzero(region & (stage == "ts"))}
    region &= np.array([b in top_of for b in band])
    top_index = np.array([top_of.get(b, 0) for b in band])
    sigma_barrier = np.where(region, energy_std[top_index], np.nan)
    sigma_barrier_all = np.where(region, energy_std_all[top_index], np.nan)
    climbing = np.flatnonzero(region & (stage == "ts"))
    score = np.full(len(frames), np.nan)
    score[region] = (sigma_max[region] / np.median(sigma_max[climbing])
                     + sigma_barrier[region] / np.median(sigma_barrier[climbing]))

    descriptors = np.load(work / "committee" / "mace-mpa-0-medium.npz")["descriptors"].astype(np.float64)
    mean_descriptor = reduce_frames(descriptors, offsets, lambda d: d.mean(axis=0))
    worst_atom = offsets[:-1] + np.array([np.argmax(sigma_atom[a:b]) for a, b in zip(offsets[:-1], offsets[1:])])
    X = np.hstack([mean_descriptor, descriptors[worst_atom]])
    X = (X - X[region].mean(axis=0)) / (X[region].std(axis=0) + 1e-12) / np.sqrt(X.shape[1])

    # Typical descriptor distance between a climbing image and its neighbouring images on the band.
    neighbour_distance = []
    for k in np.flatnonzero(region & (stage == "ts_neighbour")):
        top = np.flatnonzero((band == band[k]) & (stage == "ts"))
        if len(top):
            neighbour_distance.append(np.linalg.norm(X[k] - X[top[0]]))
    redundant_distance = float(np.median(neighbour_distance))

    candidates = np.flatnonzero(region)
    threshold = np.quantile(score[climbing], 1 - args.pool_fraction)
    pool = climbing[score[climbing] >= threshold]
    weight = score[pool] / score[pool].max()
    chosen, picked_distance = greedy_selection(X[pool], weight, band[pool], np.full(len(pool), "ts"), len(pool), 1,
                                               limits={"ts": 1})
    window = 50
    smoothed = np.array([np.median(picked_distance[max(1, k - window + 1):k + 1]) for k in range(1, len(picked_distance))])
    below = np.flatnonzero(smoothed < redundant_distance)
    natural = int(below[0]) + 1 if len(below) else len(chosen)
    n_bands = int(np.clip(natural, -(-args.min_frames // 2), min(args.max_frames // 2, len(chosen))))
    # Each chosen band adds the neighbouring or displaced image with the largest force spread.
    selected, band_order = [], []
    near = np.isin(stage, NEAR_STAGES) & region
    for n, k in enumerate(pool[chosen[:n_bands]]):
        options = np.flatnonzero(near & (band == band[k]))
        for j in [k] + ([options[np.argmax(sigma_max[options])]] if len(options) else []):
            selected.append(j)
            band_order.append(n)
    selected = np.array(selected)

    rank = np.full(len(frames), -1)
    rank[candidates[np.argsort(-score[candidates])]] = np.arange(len(candidates))
    out = []
    for k, order in zip(selected, band_order):
        atoms = frames[k].copy()
        a, b = offsets[k], offsets[k + 1]
        atoms.info.update(
            committee_force_std_max=float(sigma_max[k]), committee_force_std_mean=float(sigma_mean[k]),
            committee_barrier_std=float(sigma_barrier[k]),
            committee_barrier_mean=float(np.mean(barrier[pbe, top_index[k]])),
            committee_relative_energy_std=float(energy_std[k]),
            committee_force_std_max_with_r2scan=float(sigma_max_all[k]),
            committee_barrier_std_with_r2scan=float(sigma_barrier_all[k]),
            selection_score=float(score[k]), score_rank=int(rank[k]), selection_order=order,
            selection_coverage_distance=float(picked_distance[order]), committee_members_pbe=int(pbe.sum()),
        )
        atoms.arrays["committee_force_std"] = sigma_atom[a:b].astype(np.float64)
        atoms.arrays["committee_mean_forces"] = mean_forces[a:b]
        out.append(atoms)
    write(report / "finetune_ts_selected.extxyz", out, format="extxyz")
    # End states of every band with a selected frame, for checking barriers after labelling.
    chosen_bands = sorted({band[k] for k in selected})
    write(report / "ts_endpoints.extxyz", [frames[ends[b][s]] for b in chosen_bands for s in ("initial", "final")],
          format="extxyz")

    with gzip.open(report / "candidate_scores.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "frame_id", "pair_id", "process", "process_class", "framework", "curvature",
                         "stage", "n_atoms", "generator_barrier_forward", "force_std_max", "force_std_mean",
                         "barrier_std", "relative_energy_std", "committee_barrier_mean", "force_std_max_with_r2scan",
                         "score", "in_pool", "selected"])
        in_pool, is_selected = np.zeros(len(frames), bool), np.zeros(len(frames), bool)
        in_pool[pool] = True
        is_selected[selected] = True
        for k, f in enumerate(frames):
            i = f.info
            ts = region[k]
            writer.writerow([k, i["frame_id"], i["pair_id"], i["process"], i["process_class"], i["framework"],
                             i["curvature"], i["stage"], len(f), f"{i['barrier_forward']:.4f}",
                             f"{sigma_max[k]:.5f}", f"{sigma_mean[k]:.5f}",
                             f"{sigma_barrier[k]:.5f}" if ts else "", f"{energy_std[k]:.5f}" if ts else "",
                             f"{np.mean(barrier[pbe, top_index[k]]):.4f}" if ts else "",
                             f"{sigma_max_all[k]:.5f}", f"{score[k]:.4f}" if ts else "", int(in_pool[k]), int(is_selected[k])])
    # Each member's forward barrier on the generator's path, one row per band.
    with gzip.open(report / "barriers.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "process", "process_class", "framework", "curvature", "generator"]
                        + [m["key"] for m in members])
        for k in np.flatnonzero(region & (stage == "ts")):
            i = frames[k].info
            writer.writerow([i["pair_id"], i["process"], i["process_class"], i["framework"], i["curvature"],
                             f"{i['barrier_forward']:.4f}"] + [f"{v:.4f}" for v in barrier[:, k]])
    np.savez_compressed(report / "selection_curve.npz", picked_distance=picked_distance,
                        redundant_distance=redundant_distance, pool_scores=score[pool][chosen], n_final=n_bands, natural=natural)
    ts_frames = region & (stage == "ts")
    deviation = barrier[:, ts_frames] - np.median(barrier[pbe][:, ts_frames], axis=0)
    summary = {
        "candidates": int(len(frames)), "transition_state_frames": int(region.sum()), "bands": len(complete),
        "pool_threshold_score": float(threshold), "pool_size": int(len(pool)),
        "selected": int(len(selected)), "natural_stopping_point": natural, "redundant_distance": redundant_distance,
        "members_pbe": [m["key"] for m in members if m["pbe_level"]],
        "members_r2scan": [m["key"] for m in members if not m["pbe_level"]],
        "median_force_std_max": float(np.median(sigma_max[region])),
        "median_force_std_max_climbing": float(np.median(sigma_max[climbing])),
        "median_barrier_std": float(np.median(sigma_barrier[climbing])),
        "selected_median_force_std_max": float(np.median(sigma_max[selected])),
        "selected_median_force_std_max_climbing": float(np.median(sigma_max[pool[chosen[:n_bands]]])),
        "selected_median_barrier_std": float(np.median(sigma_barrier[pool[chosen[:n_bands]]])),
        "leave_one_out_rms_force_eV_per_A": leave_one_out_rms(forces[:, np.repeat(region, natoms)], members),
        "member_barrier_deviation_median_eV": {m["key"]: float(np.median(deviation[n])) for n, m in enumerate(members)},
        "member_barrier_deviation_mad_eV": {m["key"]: float(np.median(np.abs(deviation[n]))) for n, m in enumerate(members)},
        "selected_by_process": Counter(frames[k].info["process"] for k in selected),
        "selected_by_process_class": Counter(frames[k].info["process_class"] for k in selected),
        "selected_by_stage": Counter(stage[k] for k in selected),
        "selected_by_framework": Counter(frames[k].info["framework"] for k in selected),
        "selected_by_curvature": Counter(frames[k].info["curvature"] for k in selected),
        "selected_bands": len(chosen_bands),
        "selected_atoms": int(natoms[selected].sum()),
    }
    (report / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, (dict, list))}, indent=2))


if __name__ == "__main__":
    main()
