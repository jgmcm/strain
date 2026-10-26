#!/usr/bin/env python3
"""Score committee disagreement on the candidate pool and select the fine-tune set.

Disagreement is measured over the PBE-level members (the r2SCAN members are
reported separately, because they differ from the rest by functional as well as
by model):

- per-atom force spread: sigma_i = sqrt(mean_m |F_mi - mean_m F_mi|^2);
  per frame, its maximum and mean over atoms;
- energy spread: each member's energies minus its own least-squares elemental
  reference energies, divided by the atom count; per frame, the standard
  deviation over members (eV/atom).

The selection score adds the two spreads, each divided by its median over the
pool. Selection is a greedy farthest-point search in descriptor space, weighted
by score, over the top `--pool-fraction` of frames by score, with at most
`--per-config` frames from any one starting configuration: one unrelaxed frame
(the starting structure or a relaxation step), one relaxed minimum and two MD
frames. Each frame's
descriptor joins its mean MACE-MPA-0 atomic descriptor with the descriptor of
its most uncertain atom. Frames with elements outside `--elements` (default:
carbon only) are dropped before scoring. The final count is where the distance from each new
pick to the frames already chosen falls to the typical distance between
consecutive saved MD frames (20 fs apart), using a running median over 50
picks, bounded to the requested range.
"""

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read, write


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
ELEMENTS = ["H", "C", "N", "O", "F"]


def load_predictions(members, natoms, work):
    energies, forces = [], []
    for m in members:
        data = np.load(work / "committee" / f"{m['key']}.npz")
        assert np.array_equal(data["natoms"], natoms), f"{m['key']} was evaluated on a different pool"
        energies.append(data["energy"])
        forces.append(data["forces"].astype(np.float64))
    return np.array(energies), np.array(forces)


def reduce_frames(values, offsets, op):
    return np.array([op(values[a:b]) for a, b in zip(offsets[:-1], offsets[1:])])


def force_spread(forces, offsets):
    mean = forces.mean(axis=0)
    per_atom = np.sqrt(((forces - mean) ** 2).sum(axis=2).mean(axis=0))
    return per_atom, mean, reduce_frames(per_atom, offsets, np.max), reduce_frames(per_atom, offsets, np.mean)


def energy_residuals(energies, composition):
    """Each member's energy minus its own fitted elemental reference energies."""
    coefficients, *_ = np.linalg.lstsq(composition, energies.T, rcond=None)
    return energies - (composition @ coefficients).T, coefficients


def leave_one_out_rms(forces, members):
    """RMS force difference between each member and the mean of the other members."""
    total = forces.sum(axis=0)
    rows = {}
    for k, m in enumerate(members):
        others = (total - forces[k]) / (len(members) - 1)
        rows[m["key"]] = float(np.sqrt(((forces[k] - others) ** 2).sum(axis=1).mean()))
    return rows


STAGE_GROUP = {"initial": "unrelaxed", "relax": "unrelaxed", "relaxed": "relaxed", "md": "md"}
STAGE_LIMITS = {"unrelaxed": 1, "relaxed": 1, "md": 2}  # per starting configuration


def greedy_selection(X, weight, config_ids, stage_groups, n_max, per_config, limits=STAGE_LIMITS):
    selected, picked_distance = [], []
    counts, stage_counts = Counter(), Counter()
    first = int(np.argmax(weight))
    selected.append(first)
    picked_distance.append(np.inf)
    counts[config_ids[first]] += 1
    stage_counts[config_ids[first], stage_groups[first]] += 1
    d_min = np.linalg.norm(X - X[first], axis=1)
    available = np.ones(len(X), bool)
    available[first] = False
    while len(selected) < n_max:
        gain = np.where(available, d_min * weight, -np.inf)
        k = int(np.argmax(gain))
        if not np.isfinite(gain[k]):
            break
        available[k] = False
        if counts[config_ids[k]] >= per_config or \
                stage_counts[config_ids[k], stage_groups[k]] >= limits[stage_groups[k]]:
            continue
        selected.append(k)
        picked_distance.append(float(d_min[k]))
        counts[config_ids[k]] += 1
        stage_counts[config_ids[k], stage_groups[k]] += 1
        d_min = np.minimum(d_min, np.linalg.norm(X - X[k], axis=1))
    return np.array(selected), np.array(picked_distance)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-frames", type=int, default=1000)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--pool-fraction", type=float, default=0.4)
    parser.add_argument("--per-config", type=int, default=3)
    parser.add_argument("--work", type=Path, default=WORK, help="Directory with candidates.extxyz and committee/.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Members to leave out (preview runs).")
    parser.add_argument("--report", type=Path, default=ROOT, help="Directory for the selected frames and summaries.")
    parser.add_argument("--elements", nargs="+", default=["C"], help="Keep only frames made of these elements.")
    args = parser.parse_args()
    work, report = args.work, args.report

    committee = json.loads((ROOT / "committee.json").read_text())
    members = [m for m in committee["members"] if m["key"] not in args.exclude]
    frames = read(work / "candidates.extxyz", ":")
    natoms = np.array([len(f) for f in frames])
    energies, forces = load_predictions(members, natoms, work)
    descriptors = np.load(work / "committee" / "mace-mpa-0-medium.npz")["descriptors"].astype(np.float64)

    keep = np.array([set(f.get_chemical_symbols()) <= set(args.elements) for f in frames])
    atom_keep = np.repeat(keep, natoms)
    pool_index = np.flatnonzero(keep)  # position of each kept frame in candidates.extxyz
    excluded = Counter(f.info["defect_class"] for f, k in zip(frames, keep) if not k)
    frames, natoms = [frames[k] for k in pool_index], natoms[keep]
    energies, forces, descriptors = energies[:, keep], forces[:, atom_keep], descriptors[atom_keep]
    offsets = np.concatenate([[0], np.cumsum(natoms)])
    composition = np.array([[f.get_chemical_symbols().count(e) for e in ELEMENTS] for f in frames], float)
    composition = composition[:, composition.sum(axis=0) > 0]

    pbe = np.array([m["pbe_level"] for m in members])
    sigma_atom, mean_forces, sigma_max, sigma_mean = force_spread(forces[pbe], offsets)
    _, _, sigma_max_all, _ = force_spread(forces, offsets)
    residuals, _ = energy_residuals(energies, composition)
    per_atom_residuals = residuals / natoms
    sigma_energy = per_atom_residuals[pbe].std(axis=0)
    sigma_energy_all = per_atom_residuals.std(axis=0)
    score = sigma_max / np.median(sigma_max) + sigma_energy / np.median(sigma_energy)

    mean_descriptor = reduce_frames(descriptors, offsets, lambda d: d.mean(axis=0))
    worst_atom = offsets[:-1] + np.array([np.argmax(sigma_atom[a:b]) for a, b in zip(offsets[:-1], offsets[1:])])
    X = np.hstack([mean_descriptor, descriptors[worst_atom]])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12) / np.sqrt(X.shape[1])

    # Typical descriptor distance between consecutive saved MD frames of one trajectory.
    consecutive = []
    for k in range(len(frames) - 1):
        a, b = frames[k].info, frames[k + 1].info
        if a["stage"] == b["stage"] == "md" and a["config_id"] == b["config_id"]:
            consecutive.append(np.linalg.norm(X[k] - X[k + 1]))
    redundant_distance = float(np.median(consecutive))

    threshold = np.quantile(score, 1 - args.pool_fraction)
    pool = np.flatnonzero(score >= threshold)
    config_ids = np.array([frames[k].info["config_id"] for k in pool])
    stage_groups = np.array([STAGE_GROUP[frames[k].info["stage"]] for k in pool])
    weight = score[pool] / score[pool].max()
    chosen, picked_distance = greedy_selection(X[pool], weight, config_ids, stage_groups, args.max_frames, args.per_config)
    window = 50  # running median of pick distances, so one close pick does not end the selection
    smoothed = np.array([np.median(picked_distance[max(1, k - window + 1):k + 1]) for k in range(1, len(picked_distance))])
    below = np.flatnonzero(smoothed < redundant_distance)
    natural = int(below[0]) + 1 if len(below) else len(chosen)
    n_final = int(np.clip(natural, args.min_frames, min(args.max_frames, len(chosen))))
    selected = pool[chosen[:n_final]]

    rank = np.empty(len(frames), int)
    rank[np.argsort(-score)] = np.arange(len(frames))
    out = []
    for order, k in enumerate(selected):
        atoms = frames[k].copy()
        a, b = offsets[k], offsets[k + 1]
        atoms.info.update(
            committee_force_std_max=float(sigma_max[k]), committee_force_std_mean=float(sigma_mean[k]),
            committee_energy_std_per_atom=float(sigma_energy[k]),
            committee_force_std_max_with_r2scan=float(sigma_max_all[k]),
            committee_energy_std_per_atom_with_r2scan=float(sigma_energy_all[k]),
            selection_score=float(score[k]), score_rank=int(rank[k]), selection_order=order,
            selection_coverage_distance=float(picked_distance[order]),
            committee_members_pbe=int(pbe.sum()),
        )
        atoms.arrays["committee_force_std"] = sigma_atom[a:b].astype(np.float64)
        atoms.arrays["committee_mean_forces"] = mean_forces[a:b]
        out.append(atoms)
    write(report / "finetune_selected.extxyz", out, format="extxyz")

    with gzip.open(report / "candidate_scores.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "frame_id", "config_id", "framework", "defect_class", "curvature", "stage",
                         "generator", "md_temperature_K", "n_atoms", "force_std_max", "force_std_mean",
                         "energy_std_per_atom", "force_std_max_with_r2scan", "score", "in_pool", "selected"])
        in_pool, is_selected = np.zeros(len(frames), bool), np.zeros(len(frames), bool)
        in_pool[pool] = True
        is_selected[selected] = True
        for k, f in enumerate(frames):
            i = f.info
            writer.writerow([pool_index[k], i["frame_id"], i["config_id"], i["framework"], i["defect_class"], i["curvature"],
                             i["stage"], i["generator"], i.get("md_temperature_K", ""), len(f),
                             f"{sigma_max[k]:.5f}", f"{sigma_mean[k]:.5f}", f"{sigma_energy[k]:.6f}",
                             f"{sigma_max_all[k]:.5f}", f"{score[k]:.4f}", int(in_pool[k]), int(is_selected[k])])
    np.savez_compressed(report / "selection_curve.npz", picked_distance=picked_distance, redundant_distance=redundant_distance,
                        pool_scores=score[pool][chosen], n_final=n_final, natural=natural)
    summary = {
        "candidates": len(keep), "elements": args.elements, "excluded_frames": int((~keep).sum()),
        "excluded_by_defect_class": excluded, "scored": len(frames), "pool_threshold_score": float(threshold), "pool_size": int(len(pool)),
        "selected": n_final, "natural_stopping_point": natural, "redundant_distance": redundant_distance,
        "members_pbe": [m["key"] for m in members if m["pbe_level"]],
        "members_r2scan": [m["key"] for m in members if not m["pbe_level"]],
        "median_force_std_max": float(np.median(sigma_max)), "median_energy_std_per_atom": float(np.median(sigma_energy)),
        "selected_median_force_std_max": float(np.median(sigma_max[selected])),
        "selected_median_energy_std_per_atom": float(np.median(sigma_energy[selected])),
        "leave_one_out_rms_force_eV_per_A": leave_one_out_rms(forces, members),
        "selected_by_framework": Counter(frames[k].info["framework"] for k in selected),
        "selected_by_defect_class": Counter(frames[k].info["defect_class"] for k in selected),
        "selected_by_stage": Counter(frames[k].info["stage"] for k in selected),
        "selected_by_curvature": Counter(frames[k].info["curvature"] for k in selected),
        "selected_by_generator": Counter(frames[k].info["generator"] for k in selected),
        "selected_atoms": int(natoms[selected].sum()),
    }
    (report / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
