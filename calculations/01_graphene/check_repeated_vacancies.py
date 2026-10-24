#!/usr/bin/env python3
"""Explicitly repeat graphene cells and measure vacancy distances without PBC."""

import json
from itertools import combinations, product
from pathlib import Path

import ase
import numpy as np
from ase.build import graphene
from ase.io import read, write


OUTPUT = Path(__file__).resolve().parent
REPEATS = 7
TOL = 1e-9


def measure(vacancy_positions, center):
    """Use ordinary Cartesian distances to every other explicit vacancy site."""
    vectors = vacancy_positions - vacancy_positions[center]
    distances = np.linalg.norm(vectors, axis=1)
    distances[center] = np.inf
    order = np.argsort(distances)
    nearest_vectors = vectors[order[:3]]
    nearest_distances = distances[order[:3]]
    normalized = nearest_vectors / nearest_distances[:, None]
    angles = [
        float(np.degrees(np.arccos(np.clip(np.dot(normalized[i], normalized[j]), -1, 1))))
        for i, j in combinations(range(3), 2)
    ]
    assert np.ptp(nearest_distances) < TOL
    assert np.max(np.abs(np.array(angles) - 120)) < TOL
    assert distances[order[3]] > nearest_distances[0] + TOL
    return {
        "center_cartesian_angstrom": vacancy_positions[center].tolist(),
        "neighbor_cartesian_angstrom": vacancy_positions[order[:3]].tolist(),
        "neighbor_displacements_angstrom": nearest_vectors.tolist(),
        "neighbor_distances_angstrom": nearest_distances.tolist(),
        "pairwise_angles_degrees": angles,
        "fourth_neighbor_distance_angstrom": float(distances[order[3]]),
    }


def check(n, second):
    pristine = graphene(a=2.46, size=(n, n, 1), vacuum=15)
    if n == 4:
        np.testing.assert_allclose(
            read(OUTPUT / "graphene_4x4.extxyz").positions,
            pristine.positions, rtol=0, atol=1e-8,
        )
    # Mark actual carbon atoms before repeating the whole ASE structure.
    tags = np.zeros(len(pristine), dtype=int)
    tags[0], tags[second] = 1, 2
    pristine.set_tags(tags)
    repeated = pristine.repeat((REPEATS, REPEATS, 1))
    repeated.pbc = False
    removed = repeated.get_tags() != 0
    vacancies = repeated[removed]
    vacancies.set_chemical_symbols(["X"] * len(vacancies))
    carbon = repeated[~removed]
    assert len(vacancies) == 2 * REPEATS**2
    assert len(carbon) == (2 * n**2 - 2) * REPEATS**2

    positions = vacancies.positions
    interior_results = []
    centers = []
    # Verify both vacancy types in all 25 interior cells: 50 sites per size.
    for i, j in product(range(1, REPEATS - 1), repeat=2):
        translation = i * pristine.cell[0] + j * pristine.cell[1]
        for original_index in (0, second):
            target = pristine.positions[original_index] + translation
            center = int(np.argmin(np.linalg.norm(positions - target, axis=1)))
            assert np.linalg.norm(positions[center] - target) < TOL
            result = measure(positions, center)
            interior_results.append(result)
            if (i, j) == (3, 3):
                result["original_atom_index"] = original_index
                centers.append(result)

    stem = f"{n}x{n}_two_vacancies_repeated7x7"
    write(OUTPUT / f"graphene_{stem}.extxyz", carbon, format="extxyz")
    write(OUTPUT / f"vacancy_sites_{stem}.extxyz", vacancies, format="extxyz")
    return {
        "n": n,
        "ase_version": ase.__version__,
        "original_vacancy_indices_zero_based": [0, second],
        "repeat": [REPEATS, REPEATS, 1],
        "explicit_vacancy_site_count": len(vacancies),
        "interior_vacancy_sites_verified": len(interior_results),
        "maximum_spread_of_three_distances_angstrom": max(
            float(np.ptp(result["neighbor_distances_angstrom"]))
            for result in interior_results
        ),
        "maximum_angle_error_degrees": max(
            float(np.max(np.abs(np.array(result["pairwise_angles_degrees"]) - 120)))
            for result in interior_results
        ),
        "central_vacancy_checks": centers,
        "method": "Ordinary Cartesian distance to every vacancy in a finite 7x7 repetition; no minimum-image wrapping.",
    }


def main():
    results = [check(4, 19), check(5, 17)]
    output = OUTPUT / "explicit_repeated_vacancy_check.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        first = result["central_vacancy_checks"][0]
        print(f"{result['n']}x{result['n']}: {result['explicit_vacancy_site_count']} explicit vacancy sites;")
        print(f"  {result['interior_vacancy_sites_verified']} interior sites passed")
        print(f"  nearest distances (Angstrom): {first['neighbor_distances_angstrom']}")
        print(f"  pairwise angles (degrees): {first['pairwise_angles_degrees']}")
        print(f"  fourth neighbor (Angstrom): {first['fourth_neighbor_distance_angstrom']}")
    print(f"Full Cartesian coordinates and checks: {output}")


if __name__ == "__main__":
    main()
