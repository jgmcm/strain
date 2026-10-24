#!/usr/bin/env python3
"""Build an n x n periodic graphene monolayer and save extended XYZ."""

import argparse
import math
from pathlib import Path

from ase.build import graphene
from ase.io import write


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("n must be a positive integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "n", type=positive_int, help="Repeat the two-atom primitive cell n times in each in-plane direction."
    )
    parser.add_argument(
        "--a", type=positive_float, default=2.46, help="In-plane lattice constant in Angstrom (default: 2.46)."
    )
    parser.add_argument(
        "--vacuum", type=positive_float, default=15.0,
        help="Vacuum on each side of the sheet in Angstrom (default: 15)."
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Extended XYZ output path (default: graphene_NxN.extxyz beside this script)."
    )
    args = parser.parse_args()

    atoms = graphene(a=args.a, size=(args.n, args.n, 1), vacuum=args.vacuum)
    atoms.info.update(
        repeat_n=args.n,
        lattice_constant_angstrom=args.a,
        vacuum_per_side_angstrom=args.vacuum,
    )
    output = args.output or Path(__file__).resolve().parent / f"graphene_{args.n}x{args.n}.extxyz"
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, atoms, format="extxyz")
    print(f"Wrote {output}: {len(atoms)} C atoms, {args.n} x {args.n} primitive cells")
    print(f"a = {args.a:g} Angstrom; vacuum = {args.vacuum:g} Angstrom per side; PBC = {atoms.pbc.tolist()}")


if __name__ == "__main__":
    main()
