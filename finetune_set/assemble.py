#!/usr/bin/env python3
"""Collect the sampled frames into one candidate pool and remove unphysical frames.

A frame is removed when two atoms are closer than the minimum distances below,
when an atom has no neighbour (a desorbed or detached atom), or when the
generator's largest force exceeds 30 eV/Angstrom.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list


ROOT = Path(__file__).resolve().parent
MIN_DISTANCE = {"heavy": 1.0, "with_H": 0.8, "HH": 0.6}  # Angstrom
NEIGHBOUR_RADIUS = 2.3  # an atom with no neighbour within this distance counts as detached
MAX_GENERATOR_FORCE = 30.0  # eV/Angstrom


def problem(atoms):
    i, j, d = neighbor_list("ijd", atoms, NEIGHBOUR_RADIUS)
    h = atoms.numbers == 1
    hh, xh = h[i] & h[j], h[i] ^ h[j]
    if np.any(d[~h[i] & ~h[j]] < MIN_DISTANCE["heavy"]) or np.any(d[xh] < MIN_DISTANCE["with_H"]) or np.any(d[hh] < MIN_DISTANCE["HH"]):
        return "close contact"
    if len(np.unique(i)) < len(atoms):
        return "detached atom"
    if atoms.info["generator_fmax"] > MAX_GENERATOR_FORCE:
        return "generator force above 30 eV/A"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    files = sorted((ROOT / "work" / "sampled").glob("*/c*.extxyz"))
    kept, removed = [], Counter()
    stages = Counter()
    for path in files:
        for frame in read(path, ":"):
            reason = problem(frame)
            stages[(frame.info["generator"], frame.info["stage"])] += 1
            if reason:
                removed[(frame.info["stage"], reason)] += 1
                continue
            frame.info["frame_id"] = f"{frame.info['config_id']}-{frame.info['stage']}-{frame.info['step']}"
            frame.info["frame_index"] = len(kept)
            kept.append(frame)
    write(ROOT / "work" / "candidates.extxyz", kept, format="extxyz")
    summary = {
        "sampled_files": len(files),
        "frames_sampled": sum(stages.values()),
        "frames_kept": len(kept),
        "removed": {f"{stage}: {reason}": n for (stage, reason), n in sorted(removed.items())},
        "kept_by_stage": Counter(f.info["stage"] for f in kept),
        "kept_by_generator": Counter(f.info["generator"] for f in kept),
        "kept_by_framework": Counter(f.info["framework"] for f in kept),
        "atoms_total": int(sum(len(f) for f in kept)),
    }
    (ROOT / "candidates_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
