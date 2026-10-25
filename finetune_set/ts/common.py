"""Helpers shared by the transition-state scripts."""

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
sys.path.insert(0, str(ROOT.parent))
from build_configs import Framework  # noqa: E402

FRAMEWORK_KIND = {"sheet": "sheet", "nanotube": "tube", "fullerene": "cage"}
MIN_DISTANCE = {"heavy": 1.0, "with_H": 0.8, "HH": 0.6}  # Angstrom, as for the first set
NEIGHBOUR_RADIUS = 2.3


def load_config():
    return json.loads((ROOT / "config.json").read_text())


def event_atoms(atoms):
    """Indices of the atoms the process moves (extxyz reads the stored list as an int or an array)."""
    return np.atleast_1d(atoms.info["event_atoms"]).astype(int)


def normal(atoms, point):
    return Framework(atoms, FRAMEWORK_KIND[atoms.info["framework"]]).normal(point)


def unphysical(atoms):
    """The reason a structure is unphysical (close contact or detached atom), or None."""
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list("ijd", atoms, NEIGHBOUR_RADIUS)
    h = atoms.numbers == 1
    hh, xh = h[i] & h[j], h[i] ^ h[j]
    if np.any(d[~h[i] & ~h[j]] < MIN_DISTANCE["heavy"]) or np.any(d[xh] < MIN_DISTANCE["with_H"]) \
            or np.any(d[hh] < MIN_DISTANCE["HH"]):
        return "close contact"
    if len(np.unique(i)) < len(atoms):
        return "detached atom"
    return None


def displacement(a, b):
    """Per-atom displacement vectors from a to b under the minimum-image convention."""
    from ase.geometry import find_mic
    d, _ = find_mic(b.positions - a.positions, a.cell, a.pbc)
    return d


def base_id(band_id):
    """The pair a band comes from: follow-up bands append r, a or b to the pair_id."""
    return re.match(r"p\d+", band_id).group()
