#!/usr/bin/env python3
"""Build the initial configurations for the fine-tune candidate pool.

Each configuration combines a carbon framework (periodic graphene sheet,
nanotube or C60), point defects, strain and curvature, followed by a small
random displacement of every atom. Defects are placed on the undeformed
framework; strain and curvature are applied afterwards. Every structure has at
most 100 atoms and is periodic in all three directions with at least 12
Angstrom between periodic images.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import graphene, molecule, nanotube
from ase.geometry import find_mic
from ase.io import write
from ase.neighborlist import neighbor_list


ROOT = Path(__file__).resolve().parent
A = 2.46
BOND = A / np.sqrt(3)
SHEET_CELL_Z = 20.0
MIN_GAP = 12.0
MAX_ATOMS = 100
ADATOM_HEIGHT = {  # Angstrom above the local surface, by species and site
    "H": {"top": 1.12},
    "F": {"top": 1.50},
    "O": {"bridge": 1.27, "top": 1.50},
    "N": {"bridge": 1.45, "top": 1.55, "hollow": 1.30},
    "C": {"bridge": 1.85, "top": 1.90, "hollow": 1.55},
}
SITE_WEIGHTS = {
    "H": {"top": 1.0}, "F": {"top": 1.0}, "O": {"bridge": 0.7, "top": 0.3},
    "N": {"bridge": 0.6, "top": 0.2, "hollow": 0.2}, "C": {"bridge": 0.6, "top": 0.2, "hollow": 0.2},
}
ADATOM_SPECIES = {"C": 0.35, "H": 0.25, "O": 0.2, "N": 0.1, "F": 0.1}
RECT_SIZES = [(4, 4), (5, 4), (4, 5), (6, 3), (3, 6), (5, 5), (6, 4), (4, 6),
              (8, 3), (3, 8), (10, 2), (2, 10), (12, 2), (2, 12)]
HEX_SIZES = [4, 5, 6, 7]
TUBES = [((5, 5), 4), ((6, 6), 4), ((4, 4), 5), ((3, 3), 6), ((7, 7), 3),
         ((8, 0), 2), ((9, 0), 2), ((10, 0), 2), ((12, 0), 2)]
SHEET_DEFECTS = {  # name: (weight, operations)
    "none": (0.08, []),
    "V1": (0.10, [("vacancy", 1)]),
    "V2": (0.08, [("vacancy", 2)]),
    "V3": (0.04, [("vacancy", 3)]),
    "V4": (0.04, [("vacancy", 4)]),
    "V6 hexagon": (0.03, [("hexagon",)]),
    "V1+V1": (0.05, [("vacancy", 1), ("vacancy", 1)]),
    "V2+V1": (0.02, [("vacancy", 2), ("vacancy", 1)]),
    "SW": (0.08, [("stone_wales",)]),
    "SW+SW": (0.03, [("stone_wales",), ("stone_wales",)]),
    "adatom": (0.14, [("adatom", None)]),
    "adatom pair": (0.05, [("adatom", None), ("adatom", None)]),
    "H pair": (0.03, [("adatom", "H"), ("adatom_near", "H")]),
    "C2 dimer": (0.02, [("dimer",)]),
    "Frenkel pair": (0.04, [("vacancy", 1), ("adatom_near", "C")]),
    "decorated vacancy": (0.04, [("decorated_vacancy",)]),
    "SW+V1": (0.03, [("stone_wales",), ("vacancy", 1)]),
    "SW+adatom": (0.03, [("stone_wales",), ("adatom", None)]),
    "V2+rotated bond": (0.03, [("vacancy", 2), ("stone_wales_near",)]),
    "V1+adatom": (0.03, [("vacancy", 1), ("adatom", None)]),
}
TUBE_DEFECTS = {
    "none": (0.20, []), "V1": (0.18, [("vacancy", 1)]), "V2": (0.12, [("vacancy", 2)]),
    "SW": (0.18, [("stone_wales",)]), "adatom": (0.20, [("adatom", None)]),
    "V1+adatom": (0.06, [("vacancy", 1), ("adatom", None)]), "SW+V1": (0.06, [("stone_wales",), ("vacancy", 1)]),
}
FULLERENE_DEFECTS = {
    "none": (0.25, []), "V1": (0.2, [("vacancy", 1)]), "V2": (0.15, [("vacancy", 2)]),
    "SW": (0.2, [("stone_wales",)]), "adatom": (0.2, [("adatom", None)]),
}
CURVATURE = {"flat": 0.35, "ripple 1D": 0.25, "ripple 2D": 0.12, "bump": 0.13, "corrugation": 0.15}
STRAIN_MODES = {"none": 0.25, "uniaxial": 0.25, "biaxial": 0.20, "shear": 0.12, "general": 0.18}


def choose(rng, weights):
    keys = list(weights)
    p = np.array([weights[k][0] if isinstance(weights[k], tuple) else weights[k] for k in keys], float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def rect_sheet(nx, ny):
    s3 = np.sqrt(3)
    unit = Atoms("C4", positions=[(0, 0, 0), (A / 2, s3 * A / 6, 0), (A / 2, s3 * A / 2, 0), (0, 2 * s3 * A / 3, 0)],
                 cell=[A, s3 * A, SHEET_CELL_Z], pbc=True)
    sheet = unit.repeat((nx, ny, 1))
    sheet.positions[:, 2] = SHEET_CELL_Z / 2
    return sheet


def hex_sheet(n):
    sheet = graphene(a=A, size=(n, n, 1), vacuum=SHEET_CELL_Z / 2)
    sheet.set_cell([sheet.cell[0], sheet.cell[1], [0, 0, SHEET_CELL_Z]])
    sheet.positions[:, 2] = SHEET_CELL_Z / 2
    sheet.pbc = True
    return sheet


def tube(indices, repeats):
    t = nanotube(*indices, length=repeats, bond=BOND)
    diameter = np.ptp(t.positions[:, 0])
    t.set_cell([diameter + MIN_GAP + 4, diameter + MIN_GAP + 4, t.cell[2, 2]])
    t.center(axis=(0, 1))
    t.pbc = True
    return t


def fullerene():
    c60 = molecule("C60")
    c60.set_cell([22.0, 22.0, 22.0])
    c60.center()
    c60.pbc = True
    return c60


class Framework:
    """Geometry helpers for defect placement on a sheet, tube or cage."""

    def __init__(self, atoms, kind):
        self.atoms, self.kind = atoms, kind

    def normal(self, point):
        if self.kind == "sheet":
            return np.array([0.0, 0.0, 1.0])
        centre = self.atoms.cell.sum(axis=0) / 2
        radial = point - centre
        if self.kind == "tube":
            radial[2] = 0.0
        return radial / np.linalg.norm(radial)

    def carbon_bonds(self):
        i, j, D = neighbor_list("ijD", self.atoms, 1.75)
        carbon = self.atoms.numbers == 6
        keep = carbon[i] & carbon[j]
        return i[keep], j[keep], D[keep]

    def distance(self, p, q):
        _, d = find_mic(np.atleast_2d(q - p), self.atoms.cell, self.atoms.pbc)
        return float(d[0])

    def pick_atom(self, rng, avoid, min_sep=5.0, near=None):
        """A random framework carbon atom, away from earlier defects or close to `near`."""
        i, _, _ = self.carbon_bonds()
        threefold = np.flatnonzero(np.bincount(i, minlength=len(self.atoms)) == 3)
        pool = threefold.copy() if len(threefold) else np.flatnonzero(self.atoms.numbers == 6)
        rng.shuffle(pool)
        for separation in (min_sep, 0.6 * min_sep, 0.0):
            for k in pool:
                p = self.atoms.positions[k]
                if near is not None:
                    if 2.0 <= self.distance(near, p) <= 5.0:
                        return int(k)
                    continue
                if all(self.distance(c, p) >= separation for c in avoid):
                    return int(k)
            if near is not None:
                break
        return int(pool[0])


def op_vacancy(fw, rng, avoid, k):
    atoms = fw.atoms
    seed = fw.pick_atom(rng, avoid)
    i, j, _ = fw.carbon_bonds()
    cluster = [seed]
    while len(cluster) < k:
        frontier = [int(b) for a, b in zip(i, j) if a in cluster and b not in cluster]
        cluster.append(int(rng.choice(frontier)))
    centre = atoms.positions[seed].copy()
    del atoms[cluster]
    return centre, f"V{k}"


def op_hexagon(fw, rng, avoid):
    atoms = fw.atoms
    seed = fw.pick_atom(rng, avoid)
    i, j, D = fw.carbon_bonds()
    vectors = D[i == seed]
    centre = atoms.positions[seed] - vectors[0]
    _, d = find_mic(atoms.positions - centre, atoms.cell, atoms.pbc)
    ring = np.flatnonzero(d < 1.6)
    assert len(ring) == 6
    del atoms[ring]
    return centre, "V6"


def op_stone_wales(fw, rng, avoid, near=None):
    atoms = fw.atoms
    k = fw.pick_atom(rng, avoid, near=near)
    i, j, D = fw.carbon_bonds()
    choice = rng.choice(np.flatnonzero(i == k))
    partner, half = int(j[choice]), D[choice] / 2
    midpoint = atoms.positions[k] + half
    axis = fw.normal(midpoint)
    rotated = np.cross(axis, half) + axis * np.dot(axis, half)
    atoms.positions[k] = midpoint - rotated
    atoms.positions[partner] = midpoint + rotated
    return midpoint, "SW"


def adatom_site(fw, rng, k, site):
    i, j, D = fw.carbon_bonds()
    vectors = D[i == k]
    p = fw.atoms.positions[k]
    if site == "top" or len(vectors) < 3:
        return p
    if site == "bridge":
        return p + vectors[rng.integers(len(vectors))] / 2
    return p - vectors[rng.integers(len(vectors))]  # hexagon centre


def op_adatom(fw, rng, avoid, species=None, near=None):
    species = species or choose(rng, ADATOM_SPECIES)
    site = choose(rng, SITE_WEIGHTS[species])
    k = fw.pick_atom(rng, avoid, near=near)
    base = adatom_site(fw, rng, k, site)
    side = 1.0 if fw.kind != "sheet" or rng.random() < 0.75 else -1.0
    height = ADATOM_HEIGHT[species][site] + rng.normal(0, 0.1)
    position = base + side * height * fw.normal(base) + rng.normal(0, 0.05, 3)
    fw.atoms += Atoms(species, positions=[position])
    return base, f"{species} {site}"


def op_dimer(fw, rng, avoid):
    k = fw.pick_atom(rng, avoid)
    base = adatom_site(fw, rng, k, "hollow")
    n = fw.normal(base)
    tangent = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    tangent /= np.linalg.norm(tangent)
    centre = base + 1.6 * n
    fw.atoms += Atoms("C2", positions=[centre - 0.65 * tangent, centre + 0.65 * tangent])
    return base, "C2 dimer"


def op_decorated_vacancy(fw, rng, avoid):
    centre, _ = op_vacancy(fw, rng, avoid, 1)
    species = choose(rng, {"H": 0.3, "O": 0.3, "N": 0.25, "F": 0.15})
    position = centre + (0.6 + 0.4 * rng.random()) * fw.normal(centre) + rng.normal(0, 0.05, 3)
    fw.atoms += Atoms(species, positions=[position])
    return centre, f"V1+{species}"


def apply_defects(fw, rng, operations):
    avoid, labels = [], []
    for op in operations:
        name, *args = op
        if name == "vacancy":
            centre, label = op_vacancy(fw, rng, avoid, args[0])
        elif name == "hexagon":
            centre, label = op_hexagon(fw, rng, avoid)
        elif name == "stone_wales":
            centre, label = op_stone_wales(fw, rng, avoid)
        elif name == "stone_wales_near":
            centre, label = op_stone_wales(fw, rng, avoid, near=avoid[-1])
        elif name == "adatom":
            centre, label = op_adatom(fw, rng, avoid, args[0])
        elif name == "adatom_near":
            centre, label = op_adatom(fw, rng, avoid, args[0], near=avoid[-1])
        elif name == "dimer":
            centre, label = op_dimer(fw, rng, avoid)
        elif name == "decorated_vacancy":
            centre, label = op_decorated_vacancy(fw, rng, avoid)
        avoid.append(centre)
        labels.append(label)
    return labels


def sample_strain(rng, mode):
    eps = np.zeros((2, 2))
    if mode == "uniaxial":
        theta = rng.choice([0.0, np.pi / 2, rng.uniform(0, np.pi)])
        n = np.array([np.cos(theta), np.sin(theta)])
        magnitude = rng.uniform(-0.03, 0.15) if rng.random() < 0.8 else rng.uniform(-0.03, 0.0)
        eps = magnitude * np.outer(n, n)
    elif mode == "biaxial":
        if rng.random() < 0.5:
            eps = np.eye(2) * rng.uniform(-0.03, 0.10)
        else:
            eps = np.diag(rng.uniform(-0.02, 0.10, 2))
    elif mode == "shear":
        eps[0, 1] = eps[1, 0] = rng.uniform(-0.08, 0.08)
    elif mode == "general":
        eps = rng.uniform(-0.04, 0.08, (2, 2))
        eps = 0.5 * (eps + eps.T)
    return eps


def strain_cell(atoms, eps2):
    F = np.eye(3)
    F[:2, :2] += eps2
    atoms.set_cell(atoms.cell[:] @ F.T, scale_atoms=True)


def apply_curvature(atoms, rng, mode, eps2):
    """Out-of-plane displacement of a sheet. Ripples come with in-plane compression."""
    if mode == "flat":
        return {}
    scaled = atoms.get_scaled_positions(wrap=True)
    lengths = np.linalg.norm(atoms.cell[:2, :2], axis=1)
    if mode == "ripple 1D":
        d = int(rng.integers(2))
        k = int(rng.choice([1, 2]))
        compression = rng.uniform(0.01, 0.08)
        eps2[d, d] = -compression
        amplitude = lengths[d] * np.sqrt(compression) / (np.pi * k) * rng.uniform(0.8, 1.2)
        dz = amplitude * np.sin(2 * np.pi * k * scaled[:, d] + rng.uniform(0, 2 * np.pi))
        info = {"curvature_direction": d, "curvature_wavenumber": k, "curvature_compression": compression}
    elif mode == "ripple 2D":
        compression = rng.uniform(0.01, 0.05)
        eps2[0, 0], eps2[1, 1] = -compression, -compression
        amplitude = min(lengths) * np.sqrt(compression) / np.pi * rng.uniform(1.0, 1.5)
        dz = amplitude * np.sin(2 * np.pi * scaled[:, 0] + rng.uniform(0, 2 * np.pi)) * np.sin(
            2 * np.pi * scaled[:, 1] + rng.uniform(0, 2 * np.pi))
        info = {"curvature_compression": compression}
    elif mode == "bump":
        amplitude = rng.uniform(0.5, 2.0)
        width = rng.uniform(2.5, 4.5)
        centre = rng.random(3) @ atoms.cell
        delta = atoms.positions - centre
        delta[:, 2] = 0
        _, r = find_mic(delta, atoms.cell, [True, True, False])
        dz = amplitude * np.exp(-r**2 / (2 * width**2)) * rng.choice([-1, 1])
        info = {"curvature_width": width}
    else:  # corrugation: three random Fourier modes compatible with the cell
        dz = np.zeros(len(atoms))
        for _ in range(3):
            k1, k2 = rng.integers(0, 3, 2)
            if k1 == k2 == 0:
                k1 = 1
            dz += rng.uniform(0.05, 0.3) * np.sin(2 * np.pi * (k1 * scaled[:, 0] + k2 * scaled[:, 1]) + rng.uniform(0, 2 * np.pi))
        amplitude = float(np.abs(dz).max())
        info = {}
    atoms.positions[:, 2] += dz
    info["curvature_amplitude"] = float(amplitude)
    return info


def valid(atoms):
    if len(atoms) > MAX_ATOMS or len(atoms) == 0:
        return False
    i, j, d = neighbor_list("ijd", atoms, 1.2)
    hydrogen = (atoms.numbers[i] == 1) | (atoms.numbers[j] == 1)
    if np.any(d[~hydrogen] < 1.05) or np.any(d[hydrogen & (atoms.numbers[i] != atoms.numbers[j])] < 0.85) \
            or np.any(d[(atoms.numbers[i] == 1) & (atoms.numbers[j] == 1)] < 0.7):
        return False
    return True


def build_sheet(rng):
    defect = choose(rng, SHEET_DEFECTS)
    operations = SHEET_DEFECTS[defect][1]
    added = sum(op[0] in ("adatom", "adatom_near", "decorated_vacancy") for op in operations) + 2 * any(op[0] == "dimer" for op in operations)
    curvature = choose(rng, CURVATURE)
    rect = curvature.startswith("ripple") or rng.random() < 0.5
    sizes = [s for s in RECT_SIZES if 4 * s[0] * s[1] + added <= MAX_ATOMS] if rect else [
        n for n in HEX_SIZES if 2 * n * n + added <= MAX_ATOMS]
    if any(op[0] == "hexagon" or (op[0] == "vacancy" and op[1] >= 3) for op in operations):
        # Large vacancy clusters need room: at least 64 atoms and 9.5 Angstrom in both directions.
        if rect:
            sizes = [s for s in sizes if 4 * s[0] * s[1] >= 64 and min(s[0] * A, s[1] * A * np.sqrt(3)) >= 9.5]
        else:
            sizes = [s for s in sizes if 2 * s * s >= 64]
    size = sizes[rng.integers(len(sizes))]
    atoms = rect_sheet(*size) if rect else hex_sheet(size)
    fw = Framework(atoms, "sheet")
    labels = apply_defects(fw, rng, operations)
    atoms = fw.atoms
    atoms.wrap()
    strain_mode = choose(rng, STRAIN_MODES)
    eps2 = sample_strain(rng, strain_mode)
    curvature_info = apply_curvature(atoms, rng, curvature, eps2)
    strain_cell(atoms, eps2)
    z = atoms.positions[:, 2]
    if np.ptp(z) + MIN_GAP > atoms.cell[2, 2]:
        atoms.cell[2, 2] = np.ptp(z) + MIN_GAP
    atoms.positions[:, 2] += atoms.cell[2, 2] / 2 - (z.max() + z.min()) / 2
    info = {
        "framework": "sheet", "framework_detail": f"rect {size[0]}x{size[1]}" if rect else f"hex {size}x{size}",
        "defect_class": defect, "defects": ";".join(labels) or "none",
        "strain_mode": strain_mode if curvature not in ("ripple 1D", "ripple 2D") else f"{strain_mode}+compression",
        "strain_xx": float(eps2[0, 0]), "strain_yy": float(eps2[1, 1]), "strain_xy": float(eps2[0, 1]),
        "curvature": curvature,
    }
    info.update(curvature_info)
    return atoms, info


def build_tube(rng):
    indices, repeats = TUBES[rng.integers(len(TUBES))]
    defect = choose(rng, TUBE_DEFECTS)
    atoms = tube(indices, repeats)
    fw = Framework(atoms, "tube")
    labels = apply_defects(fw, rng, TUBE_DEFECTS[defect][1])
    atoms = fw.atoms
    axial = rng.uniform(-0.03, 0.12) if rng.random() < 0.7 else 0.0
    ellipticity = rng.uniform(0.0, 0.15) if rng.random() < 0.4 else 0.0
    centre = atoms.cell.sum(axis=0) / 2
    atoms.positions[:, 0] = centre[0] + (atoms.positions[:, 0] - centre[0]) * (1 + ellipticity)
    atoms.positions[:, 1] = centre[1] + (atoms.positions[:, 1] - centre[1]) * (1 - ellipticity)
    atoms.set_cell(np.diag([atoms.cell[0, 0], atoms.cell[1, 1], atoms.cell[2, 2] * (1 + axial)]), scale_atoms=False)
    atoms.positions[:, 2] *= 1 + axial
    atoms.wrap()
    return atoms, {
        "framework": "nanotube", "framework_detail": f"({indices[0]},{indices[1]}) x{repeats}",
        "defect_class": defect, "defects": ";".join(labels) or "none",
        "strain_mode": "axial" if axial else "none", "strain_axial": float(axial),
        "curvature": "tube", "ellipticity": float(ellipticity),
    }


def build_fullerene(rng):
    defect = choose(rng, FULLERENE_DEFECTS)
    atoms = fullerene()
    fw = Framework(atoms, "cage")
    labels = apply_defects(fw, rng, FULLERENE_DEFECTS[defect][1])
    atoms = fw.atoms
    squash = rng.uniform(0.0, 0.08) if rng.random() < 0.4 else 0.0
    scale = 1 + (rng.uniform(-0.03, 0.03) if rng.random() < 0.5 else 0.0)
    centre = atoms.cell.sum(axis=0) / 2
    atoms.positions = centre + (atoms.positions - centre) * scale * np.array([1, 1, 1 - squash])
    return atoms, {
        "framework": "fullerene", "framework_detail": "C60",
        "defect_class": defect, "defects": ";".join(labels) or "none",
        "strain_mode": "isotropic+squash" if squash or scale != 1 else "none",
        "strain_isotropic": float(scale - 1), "squash": float(squash), "curvature": "cage",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-configs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "initial.extxyz")
    args = parser.parse_args()
    generators = json.loads((ROOT / "committee.json").read_text())["generators"]
    shares = {key: spec["share"] for key, spec in generators.items()}
    rng = np.random.default_rng(args.seed)
    builders = {"sheet": (0.78, build_sheet), "nanotube": (0.15, build_tube), "fullerene": (0.07, build_fullerene)}
    frames = []
    while len(frames) < args.n_configs:
        framework = choose(rng, {k: v[0] for k, v in builders.items()})
        for _ in range(50):
            atoms, info = builders[framework][1](rng)
            noise = rng.uniform(0.01, 0.05)
            atoms.positions += rng.normal(0, noise, atoms.positions.shape)
            if valid(atoms):
                break
        else:
            raise RuntimeError(f"Could not build a valid {framework} configuration")
        index = len(frames)
        info.update(config_id=f"c{index:05d}", noise_sigma=float(noise), n_atoms=len(atoms),
                    formula=atoms.get_chemical_formula(), generator=choose(rng, shares))
        atoms.info.update(info)
        frames.append(atoms)
    args.output.parent.mkdir(exist_ok=True)
    write(args.output, frames, format="extxyz")
    summary = {
        "n_configs": len(frames), "seed": args.seed,
        "framework": Counter(f.info["framework"] for f in frames),
        "defect_class": Counter(f.info["defect_class"] for f in frames),
        "curvature": Counter(f.info["curvature"] for f in frames),
        "strain_mode": Counter(f.info["strain_mode"] for f in frames),
        "generator": Counter(f.info["generator"] for f in frames),
        "n_atoms_min_max": [min(len(f) for f in frames), max(len(f) for f in frames)],
        "elements": Counter(s for f in frames for s in set(f.get_chemical_symbols())),
    }
    (ROOT / "configs_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
