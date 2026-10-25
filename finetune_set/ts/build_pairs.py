#!/usr/bin/env python3
"""Build initial and final states for NEB calculations of point-defect processes.

Each pair is one elementary step on a graphene sheet, nanotube or C60 cage: a
vacancy hop, a bond rotation, an adatom hop, a Frenkel-pair recombination, and
so on. Both states contain the same atoms in the same order, and differ in the
positions of the few atoms the process moves (`event_atoms`). The process is set
up on the undeformed framework; both states then receive the same strain and
curvature. Every structure has at most 100 atoms and is periodic in all three
directions with at least 12 Angstrom between periodic images.

Output: work/pairs.extxyz with the initial and final state of each pair, in
that order.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import write

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from build_configs import (ADATOM_HEIGHT, A, HEX_SIZES, MAX_ATOMS, MIN_GAP, SHEET_CELL_Z,  # noqa: E402
                           STRAIN_MODES, TUBES, Framework, apply_curvature, choose, fullerene, hex_sheet, op_adatom,
                           op_stone_wales, op_vacancy, rect_sheet, sample_strain, strain_cell, tube, valid)

ADATOM_SPECIES = {"C": 0.35, "H": 0.2, "O": 0.2, "N": 0.1, "F": 0.15}
HOP_SITE = {"C": {"bridge": 1.0}, "O": {"bridge": 1.0}, "N": {"bridge": 0.7, "top": 0.3}, "H": {"top": 1.0},
            "F": {"top": 1.0}}
SHEET_CURVATURE = {"flat": 0.45, "ripple 1D": 0.35, "ripple 2D": 0.20}
# Rectangular cells of 4-atom units with 48-80 atoms: smaller on average than in the first set, because
# NEB cost grows with the number of atoms times the number of images.
TS_RECT_SIZES = [(3, 4), (4, 3), (6, 2), (2, 6), (3, 5), (5, 3), (4, 4), (8, 2), (2, 8), (5, 4), (4, 5), (6, 3),
                 (3, 6), (10, 2), (2, 10)]


def without(atoms, removed):
    """Atoms with `removed` deleted, and the map from old to new indices."""
    keep = np.setdiff1d(np.arange(len(atoms)), removed)
    index = -np.ones(len(atoms), int)
    index[keep] = np.arange(len(keep))
    return atoms[keep], index


def bonds_of(fw, k):
    i, j, D = fw.carbon_bonds()
    return j[i == k], D[i == k]


def near_image(fw, point, reference):
    """The periodic image of `point` closest to `reference`."""
    d, _ = find_mic(np.atleast_2d(point - reference), fw.atoms.cell, fw.atoms.pbc)
    return reference + d[0]


def lift(fw, site, species, kind, side):
    return site + side * ADATOM_HEIGHT[species][kind] * fw.normal(site)


def side_for(fw, rng):
    return 1.0 if fw.kind != "sheet" or rng.random() < 0.75 else -1.0


# Processes. Each returns (initial, final, event_atoms, centre, detail) for the framework in `fw`, where
# `detail` names what moves.

def vacancy_hop(fw, rng, size):
    """A carbon atom next to a vacancy cluster of `size` sites hops into one of the vacant sites."""
    i, j, D = fw.carbon_bonds()
    seed = fw.pick_atom(rng, [])
    cluster = [seed]
    while len(cluster) < size:
        frontier = [int(b) for a, b in zip(i, j) if a in cluster and b not in cluster]
        cluster.append(int(rng.choice(frontier)))
    options = [n for n in range(len(i)) if i[n] in cluster and j[n] not in cluster]
    n = options[rng.integers(len(options))]
    site, mover = int(i[n]), int(j[n])
    target = fw.atoms.positions[mover] - D[n]
    initial, index = without(fw.atoms, cluster)
    final = initial.copy()
    final.positions[index[mover]] = target
    return initial, final, [index[mover]], fw.atoms.positions[site], f"V{size}"


def vacancy_coalescence(fw, rng):
    """Two vacancies on second-neighbour sites; their common neighbour hops into one, leaving a divacancy."""
    i, j, D = fw.carbon_bonds()
    v1 = fw.pick_atom(rng, [])
    n = rng.choice(np.flatnonzero(i == v1))
    common, to_common = int(j[n]), D[n]
    m = rng.choice([m for m in np.flatnonzero(i == common) if j[m] != v1])
    v2 = int(j[m])
    initial, index = without(fw.atoms, [v1, v2])
    final = initial.copy()
    final.positions[index[common]] = fw.atoms.positions[common] - to_common
    return initial, final, [index[common]], fw.atoms.positions[v1], "V1+V1"


def vacancy_switch(fw, rng):
    """Monovacancy: the reconstructed bond switches from one pair of under-coordinated atoms to another."""
    seed = fw.pick_atom(rng, [])
    neighbours, vectors = bonds_of(fw, seed)
    order = rng.permutation(len(neighbours))[:3]
    (a, b, c), (va, vb, vc) = neighbours[order], vectors[order]
    initial, index = without(fw.atoms, [seed])
    final = initial.copy()
    for state, (p, q, vp, vq) in ((initial, (a, b, va, vb)), (final, (b, c, vb, vc))):
        u = (vq - vp) / np.linalg.norm(vq - vp)
        state.positions[index[p]] += 0.3 * u
        state.positions[index[q]] -= 0.3 * u
    return initial, final, [index[a], index[b], index[c]], fw.atoms.positions[seed], "V1"


def bond_rotation(fw, rng, centre=None, radius=3.5):
    """A C-C bond rotates by 90 degrees about its midpoint and the local surface normal."""
    i, j, D = fw.carbon_bonds()
    positions = fw.atoms.positions
    candidates = np.flatnonzero(i < j)
    if centre is not None:
        mids = positions[i[candidates]] + D[candidates] / 2
        candidates = candidates[[fw.distance(centre, m) <= radius for m in mids]]
    n = int(rng.choice(candidates))
    k, p, half = int(i[n]), int(j[n]), D[n] / 2
    midpoint = positions[k] + half
    axis = fw.normal(midpoint)
    rotated = np.cross(axis, half) + axis * np.dot(axis, half)
    initial = fw.atoms.copy()
    final = initial.copy()
    final.positions[k] = midpoint - rotated
    final.positions[p] = near_image(fw, midpoint + rotated, positions[p])  # p may sit across a cell boundary
    return initial, final, [k, p], midpoint, "C-C bond"


def adatom_hop(fw, rng, species, near=None):
    """An adatom moves to a neighbouring site of the same kind: bridge to bridge or top to top."""
    kind = choose(rng, HOP_SITE[species])
    k = fw.pick_atom(rng, [], near=near)
    _, vectors = bonds_of(fw, k)
    p = fw.atoms.positions[k]
    a, b = rng.choice(len(vectors), 2, replace=False)
    if kind == "top":
        start, end = p, p + vectors[a]
    else:
        start, end = p + vectors[a] / 2, p + vectors[b] / 2
    side = side_for(fw, rng)
    initial = fw.atoms + Atoms(species, positions=[lift(fw, start, species, kind, side)])
    final = fw.atoms + Atoms(species, positions=[lift(fw, end, species, kind, side)])
    return initial, final, [len(initial) - 1], start, f"{species} {kind}"


def recombination(fw, rng, species):
    """An adatom 1.5-4.5 Angstrom from a monovacancy moves into it.

    For carbon this is Frenkel-pair recombination; for H, N, O or F the adatom ends at the vacancy.
    """
    i, j, D = fw.carbon_bonds()
    v = fw.pick_atom(rng, [])
    vacancy = fw.atoms.positions[v]
    kind = choose(rng, HOP_SITE[species]) if species != "C" else "bridge"
    keep = (i < j) & (i != v) & (j != v)
    sites = fw.atoms.positions[i[keep]] + (D[keep] / 2 if kind == "bridge" else 0.0)
    sites = sites[[1.5 <= fw.distance(vacancy, s) <= 4.5 for s in sites]]
    start = sites[rng.integers(len(sites))]
    side = side_for(fw, rng)
    host, index = without(fw.atoms, [v])
    target = near_image(fw, vacancy, start)
    if species == "C":
        end = target
    elif species in ("N", "O"):
        end = target + side * 0.5 * fw.normal(target)
    else:  # H and F bind to one of the under-coordinated neighbours
        neighbours, vectors = bonds_of(fw, v)
        n = rng.integers(len(neighbours))
        atom = target + vectors[n]
        end = atom - (1.10 if species == "H" else 1.40) * vectors[n] / np.linalg.norm(vectors[n]) \
            + side * 0.4 * fw.normal(atom)
    initial = host + Atoms(species, positions=[lift(fw, start, species, kind, side)])
    final = host + Atoms(species, positions=[end])
    return initial, final, [len(initial) - 1], vacancy, f"V1+{species}"


def with_context(setup, process):
    """Apply a defect first (`setup` returns its centre and label), then run `process` near it.

    Returns the process result followed by the label of the defect, the `context`.
    """
    def run(fw, rng):
        centre, label = setup(fw, rng)
        return (*process(fw, rng, centre), label)
    return run


def rotation_near(centre_fn):
    return with_context(centre_fn, lambda fw, rng, c: bond_rotation(fw, rng, centre=c))


def vacancy_cluster(size):
    def setup(fw, rng):
        return op_vacancy(fw, rng, [], size)
    return setup


def stone_wales(fw, rng):
    return op_stone_wales(fw, rng, [])


def adatom(species=None):
    def setup(fw, rng):
        return op_adatom(fw, rng, [], species)
    return setup


def hop_species(rng):
    return choose(rng, ADATOM_SPECIES)


# name: (weight, process class, function, interior images)
SHEET_PROCESSES = {
    "V1 hop": (0.11, "vacancy migration", lambda fw, rng: vacancy_hop(fw, rng, 1), 5),
    "V2 hop": (0.05, "vacancy migration", lambda fw, rng: vacancy_hop(fw, rng, 2), 5),
    "V3/V4 hop": (0.04, "vacancy migration", lambda fw, rng: vacancy_hop(fw, rng, int(rng.choice([3, 4]))), 5),
    "V1+V1 coalescence": (0.05, "vacancy migration", vacancy_coalescence, 5),
    "V1 reconstruction switch": (0.04, "vacancy reconstruction", vacancy_switch, 5),
    "SW formation": (0.10, "bond rotation", bond_rotation, 7),
    "bond rotation at V1": (0.04, "bond rotation", rotation_near(vacancy_cluster(1)), 7),
    "bond rotation at V2": (0.07, "bond rotation", rotation_near(vacancy_cluster(2)), 7),
    "bond rotation at V3/V4": (0.03, "bond rotation", lambda fw, rng: rotation_near(vacancy_cluster(int(rng.choice([3, 4]))))(fw, rng), 7),
    "bond rotation at SW": (0.04, "bond rotation", rotation_near(stone_wales), 7),
    "bond rotation at adatom": (0.03, "bond rotation", rotation_near(adatom()), 7),
    "adatom hop": (0.20, "adatom migration", lambda fw, rng: adatom_hop(fw, rng, hop_species(rng)), 5),
    "adatom hop near vacancy": (0.03, "adatom migration",
                                with_context(vacancy_cluster(1), lambda fw, rng, c: adatom_hop(fw, rng, hop_species(rng), near=c)), 5),
    "C adatom hop near C adatom": (0.03, "adatom migration",
                                   with_context(adatom("C"), lambda fw, rng, c: adatom_hop(fw, rng, "C", near=c)), 5),
    "Frenkel recombination": (0.06, "recombination", lambda fw, rng: recombination(fw, rng, "C"), 7),
    "adatom capture at vacancy": (0.05, "recombination",
                                  lambda fw, rng: recombination(fw, rng, choose(rng, {"H": .3, "O": .3, "N": .25, "F": .15})), 7),
}
TUBE_PROCESSES = {
    "V1 hop": (0.22, "vacancy migration", lambda fw, rng: vacancy_hop(fw, rng, 1), 5),
    "V1+V1 coalescence": (0.05, "vacancy migration", vacancy_coalescence, 5),
    "V1 reconstruction switch": (0.05, "vacancy reconstruction", vacancy_switch, 5),
    "SW formation": (0.20, "bond rotation", bond_rotation, 7),
    "bond rotation at V2": (0.08, "bond rotation", rotation_near(vacancy_cluster(2)), 7),
    "adatom hop": (0.28, "adatom migration", lambda fw, rng: adatom_hop(fw, rng, hop_species(rng)), 5),
    "Frenkel recombination": (0.12, "recombination", lambda fw, rng: recombination(fw, rng, "C"), 7),
}
CAGE_PROCESSES = {
    "SW formation": (0.35, "bond rotation", bond_rotation, 7),
    "V1 hop": (0.20, "vacancy migration", lambda fw, rng: vacancy_hop(fw, rng, 1), 5),
    "adatom hop": (0.30, "adatom migration", lambda fw, rng: adatom_hop(fw, rng, hop_species(rng)), 5),
    "Frenkel recombination": (0.15, "recombination", lambda fw, rng: recombination(fw, rng, "C"), 7),
}
ROTATIONS = {"bond rotation"}


def sheet_host(rng, process):
    curvature = choose(rng, SHEET_CURVATURE)
    rect = curvature != "flat" or rng.random() < 0.5
    big = process in ("V3/V4 hop", "bond rotation at V3/V4")
    if rect:
        sizes = TS_RECT_SIZES
        if big:
            sizes = [s for s in sizes if 4 * s[0] * s[1] >= 64 and min(s[0] * A, s[1] * A * np.sqrt(3)) >= 9.5]
        size = sizes[rng.integers(len(sizes))]
        return rect_sheet(*size), f"rect {size[0]}x{size[1]}", curvature
    sizes = [n for n in HEX_SIZES if 2 * n * n <= MAX_ATOMS - 2 and (not big or 2 * n * n >= 64)]
    n = sizes[rng.integers(len(sizes))]
    return hex_sheet(n), f"hex {n}x{n}", curvature


def deform_sheet(states, rng, curvature):
    """The same strain and curvature for both states; returns the information for the frames."""
    strain_mode = choose(rng, STRAIN_MODES)
    eps2 = sample_strain(rng, strain_mode)
    seed = int(rng.integers(2**32))
    for state in states:
        e = eps2.copy()
        curvature_info = apply_curvature(state, np.random.default_rng(seed), curvature, e)
        strain_cell(state, e)
    z = np.concatenate([s.positions[:, 2] for s in states])
    height = max(SHEET_CELL_Z, np.ptp(z) + MIN_GAP)
    for state in states:
        state.cell[2, 2] = height
        state.positions[:, 2] += height / 2 - (z.max() + z.min()) / 2
    info = {"strain_mode": strain_mode if curvature == "flat" else f"{strain_mode}+compression",
            "strain_xx": float(e[0, 0]), "strain_yy": float(e[1, 1]), "strain_xy": float(e[0, 1]), "curvature": curvature}
    info.update(curvature_info)
    return info


def deform_tube(states, rng):
    axial = rng.uniform(-0.03, 0.12) if rng.random() < 0.7 else 0.0
    ellipticity = rng.uniform(0.0, 0.15) if rng.random() < 0.4 else 0.0
    for state in states:
        centre = state.cell.sum(axis=0) / 2
        state.positions[:, 0] = centre[0] + (state.positions[:, 0] - centre[0]) * (1 + ellipticity)
        state.positions[:, 1] = centre[1] + (state.positions[:, 1] - centre[1]) * (1 - ellipticity)
        state.set_cell(np.diag([state.cell[0, 0], state.cell[1, 1], state.cell[2, 2] * (1 + axial)]), scale_atoms=False)
        state.positions[:, 2] *= 1 + axial
    return {"strain_mode": "axial" if axial else "none", "strain_axial": float(axial), "curvature": "tube",
            "ellipticity": float(ellipticity)}


def build_pair(rng, framework):
    if framework == "sheet":
        process = choose(rng, SHEET_PROCESSES)
        _, kind, function, n_images = SHEET_PROCESSES[process]
        host, framework_detail, curvature = sheet_host(rng, process)
        fw = Framework(host, "sheet")
    elif framework == "nanotube":
        process = choose(rng, TUBE_PROCESSES)
        _, kind, function, n_images = TUBE_PROCESSES[process]
        indices, repeats = TUBES[rng.integers(len(TUBES))]
        fw = Framework(tube(indices, repeats), "tube")
        framework_detail = f"({indices[0]},{indices[1]}) x{repeats}"
    else:
        process = choose(rng, CAGE_PROCESSES)
        _, kind, function, n_images = CAGE_PROCESSES[process]
        fw = Framework(fullerene(), "cage")
        framework_detail = "C60"
    initial, final, event, centre, detail, *context = function(fw, rng)
    states = [initial, final]
    if framework == "sheet":
        info = deform_sheet(states, rng, curvature)
    elif framework == "nanotube":
        info = deform_tube(states, rng)
    else:
        info = {"strain_mode": "none", "curvature": "cage"}
    info.update(process=process, process_class=kind, framework=framework, framework_detail=framework_detail,
                detail=detail, context=context[0] if context else "none", event_atoms=" ".join(str(int(k)) for k in event), n_images=n_images,
                rotation=int(kind in ROTATIONS))
    return states, info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "pairs.extxyz")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    frameworks = {"sheet": 0.75, "nanotube": 0.17, "fullerene": 0.08}
    frames, records = [], []
    while len(records) < args.n_pairs:
        framework = choose(rng, frameworks)
        for _ in range(50):
            try:
                states, info = build_pair(rng, framework)
            except (ValueError, IndexError):  # no suitable site in this cell; draw again
                continue
            if all(valid(s) for s in states):
                break
        else:
            raise RuntimeError(f"Could not build a valid {framework} pair")
        pair_id = f"p{len(records):05d}"
        for state, label in zip(states, ("initial", "final")):
            state.info.update(info, pair_id=pair_id, state=label, n_atoms=len(state), formula=state.get_chemical_formula())
            frames.append(state)
        records.append(info)
    args.output.parent.mkdir(exist_ok=True)
    write(args.output, frames, format="extxyz")
    summary = {
        "n_pairs": len(records), "seed": args.seed,
        "framework": Counter(r["framework"] for r in records),
        "process_class": Counter(r["process_class"] for r in records),
        "process": Counter(f"{r['framework']}: {r['process']}" for r in records),
        "curvature": Counter(r["curvature"] for r in records),
        "strain_mode": Counter(r["strain_mode"] for r in records),
        "n_atoms_min_max": [min(len(f) for f in frames), max(len(f) for f in frames)],
        "n_atoms_mean": float(np.mean([len(f) for f in frames])),
    }
    (ROOT / "pairs_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
