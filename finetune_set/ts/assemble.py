#!/usr/bin/env python3
"""Collect the transition-state region of every usable band into the candidate pool.

A band is usable when its climbing-image NEB converged (or ended with a largest
NEB force below the acceptance limit) and its climbing image lies at least
`min_barrier_eV` (config.json) above the higher end state. From each usable
band the pool takes:

- the climbing image (`ts`);
- the images on either side of it (`ts_neighbour`), when they are interior images;
- copies of the climbing image with every atom displaced by a Gaussian of
  standard deviation 0.02-0.05 Angstrom (`ts_displaced`);
- the two end states (`initial`, `final`), which the committee evaluates so that
  each member's barrier can be computed. They are not candidates for selection.

Frames with close contacts or detached atoms are dropped.
"""

import argparse
import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read, write

from common import WORK, base_id, load_config, unphysical

USABLE = {"converged", "converged (loose)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    config = load_config()["ts_region"]
    min_barrier = load_config()["neb"]["min_barrier_eV"]
    records = {json.loads(line)["pair_id"]: json.loads(line) for line in (args.work / "neb.jsonl").open()}
    bands = {}
    for frame in read(args.work / "bands.extxyz", ":"):
        bands.setdefault(frame.info["pair_id"], []).append(frame)
    # Pair metadata comes from pairs.extxyz, the record of how each pair was built.
    pairs = {f.info["pair_id"]: f.info for f in read(args.work / "pairs.extxyz", "::2")}
    kept, removed = [], Counter()

    def add(frame, stage, image):
        frame = frame.copy()
        reason = unphysical(frame)
        if reason:
            removed[f"{stage}: {reason}"] += 1
            return
        frame.info.update(stage=stage, frame_id=f"{frame.info['pair_id']}-{stage}-{image}", frame_index=len(kept),
                          framework_detail=pairs[base_id(frame.info["pair_id"])]["framework_detail"])
        kept.append(frame)

    for pair_id, images in bands.items():
        record = records[pair_id]
        e = record["energies"]
        if record["status"] not in USABLE or e[record["climbing_image"]] - max(e[0], e[-1]) < min_barrier:
            continue
        images = sorted(images, key=lambda f: f.info["image"])
        top = record["climbing_image"]
        add(images[0], "initial", 0)
        add(images[-1], "final", len(images) - 1)
        add(images[top], "ts", top)
        for m in (top - 1, top + 1):
            if 0 < m < len(images) - 1:
                add(images[m], "ts_neighbour", m)
        rng = np.random.default_rng(20261025 + zlib.crc32(pair_id.encode()))
        for copy in range(config["displaced_copies"]):
            frame = images[top].copy()
            sigma = rng.uniform(*config["displacement_sigma_angstrom"])
            frame.positions += rng.normal(0, sigma, frame.positions.shape)
            frame.info["displacement_sigma"] = float(sigma)
            add(frame, "ts_displaced", f"{top}d{copy}")
    write(args.work / "candidates.extxyz", kept, format="extxyz")
    summary = {
        "bands": len(bands), "bands_by_status": Counter(r["status"] for r in records.values()),
        "usable_bands": len({f.info["pair_id"] for f in kept}),
        "frames": len(kept), "frames_by_stage": Counter(f.info["stage"] for f in kept),
        "removed": dict(removed), "atoms_total": int(sum(len(f) for f in kept)),
    }
    (args.work.parent / "candidates_summary.json" if args.work == WORK else args.work / "candidates_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
