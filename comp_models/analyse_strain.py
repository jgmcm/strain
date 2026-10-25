#!/usr/bin/env python3
"""Compute atomic strain, bond strain and vacancy geometry for every converged run.

Per-atom strain uses the least-squares in-plane deformation gradient F that maps
each atom's same-sublattice (second-neighbour, reference length a) vectors from
the ideal fixed-cell structure onto the relaxed structure. Same-sublattice
vectors exclude the internal shift between the two graphene sublattices.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ase.build import graphene
from ase.geometry import find_mic
from ase.io import read
from ase.neighborlist import neighbor_list


ROOT = Path(__file__).resolve().parent
FIRST_SHELL = (1.0, 1.8)
SECOND_SHELL = (2.0, 2.7)
SIZES = (4, 5)
RECONSTRUCTION_SPREAD = 0.1  # Angstrom between longest and shortest vacancy-neighbour distance


def shell(reference, bounds):
    i, j, D, S = neighbor_list("ijDS", reference, bounds[1])
    keep = np.linalg.norm(D, axis=1) > bounds[0]
    return i[keep], j[keep], D[keep], S[keep]


def current_vectors(reference, displacement, pairs, cell=None):
    i, j, D, S = pairs
    shift = 0 if cell is None else S @ (np.asarray(cell) - np.asarray(reference.cell))
    return D + displacement[j] - displacement[i] + shift


def atomic_strain(reference, displacement, cell=None):
    pairs = shell(reference, SECOND_SHELL)
    i = pairs[0]
    d0 = pairs[2][:, :2]
    d = current_vectors(reference, displacement, pairs, cell)[:, :2]
    count = np.bincount(i, minlength=len(reference))
    A = np.zeros((len(reference), 2, 2))
    B = np.zeros((len(reference), 2, 2))
    np.add.at(A, i, d0[:, :, None] * d0[:, None, :])
    np.add.at(B, i, d[:, :, None] * d0[:, None, :])
    F = B @ np.linalg.inv(A)
    E = 0.5 * (np.transpose(F, (0, 2, 1)) @ F - np.eye(2))
    principal = np.linalg.eigvalsh(E)
    return {
        "second_neighbours": count,
        "F": F,
        "E_xx": E[:, 0, 0], "E_yy": E[:, 1, 1], "E_xy": E[:, 0, 1],
        "areal_strain": np.linalg.det(F) - 1,
        "principal_max": principal[:, 1], "principal_min": principal[:, 0],
        "max_shear_strain": 0.5 * (principal[:, 1] - principal[:, 0]),
    }


def bond_strain(reference, displacement):
    pairs = shell(reference, FIRST_SHELL)
    i, j = pairs[:2]
    unique = i < j
    reference_length = np.linalg.norm(pairs[2], axis=1)
    relaxed_length = np.linalg.norm(current_vectors(reference, displacement, pairs), axis=1)
    midpoint = reference.positions[i, :2] + 0.5 * pairs[2][:, :2]
    return {
        "i": i[unique], "j": j[unique], "x_mid_ref": midpoint[unique, 0], "y_mid_ref": midpoint[unique, 1],
        "reference_length": reference_length[unique], "relaxed_length": relaxed_length[unique],
        "strain": relaxed_length[unique] / reference_length[unique] - 1,
    }


def vacancy_geometry(reference, displacement, vacancies):
    triangles = []
    for site in vacancies.positions:
        _, distances = find_mic(reference.positions - site, reference.cell, reference.pbc)
        neighbours = np.flatnonzero(distances < FIRST_SHELL[1])
        assert len(neighbours) == 3, "Each vacancy must have three surviving first neighbours"
        lengths = []
        for p, q in ((0, 1), (0, 2), (1, 2)):
            a, b = neighbours[p], neighbours[q]
            vector, _ = find_mic(reference.positions[b] - reference.positions[a], reference.cell, reference.pbc)
            lengths.append(float(np.linalg.norm(vector + displacement[b] - displacement[a])))
        triangles.append({"atoms": neighbours.tolist(), "pair_distances_angstrom": sorted(lengths)})
    return triangles


def self_test():
    """Check the strain estimator on zero and homogeneous deformations of a vacancy cell."""
    pristine = graphene(a=2.46, size=(4, 4, 1), vacuum=15.0)
    keep = np.ones(len(pristine), dtype=bool)
    keep[[0, 19]] = False
    reference = pristine[keep]
    zero = atomic_strain(reference, np.zeros((len(reference), 3)))
    F0 = np.array([[1.012, 0.004, 0.0], [-0.003, 0.991, 0.0], [0.0, 0.0, 1.0]])
    deformed_cell = np.asarray(reference.cell) @ F0.T
    displacement = reference.positions @ F0.T - reference.positions
    affine = atomic_strain(reference, displacement, cell=deformed_cell)
    affine_bonds = bond_strain(reference, np.zeros((len(reference), 3)))
    return {
        "zero_displacement_max_abs_strain": float(np.abs(zero["areal_strain"]).max()),
        "homogeneous_F_max_error": float(np.abs(affine["F"] - F0[:2, :2]).max()),
        "zero_displacement_max_abs_bond_strain": float(np.abs(affine_bonds["strain"]).max()),
        "second_neighbour_counts": sorted(set(zero["second_neighbours"].tolist())),
    }


def analyse(model, n):
    directory = ROOT / "runs" / model["key"] / f"{n}x{n}"
    metrics = json.loads((directory / "metrics.json").read_text())
    if not (metrics["converged"] and metrics["fresh_evaluation"]["passed"]):
        raise RuntimeError(f"{directory} is not a converged, validated run")
    reference = read(directory / "reference.extxyz")
    relaxed = read(directory / "relaxed.extxyz")
    vacancies = read(directory / "vacancy_sites.extxyz")
    np.testing.assert_array_equal(reference.arrays["original_site_index"], relaxed.arrays["original_site_index"])
    displacement, _ = find_mic(relaxed.positions - reference.positions, reference.cell, reference.pbc)
    displacement -= displacement.mean(axis=0)
    saved = np.genfromtxt(directory / "displacement.csv", delimiter=",", names=True)
    csv_error = float(np.abs(np.column_stack([saved["ux"], saved["uy"], saved["uz"]]) - displacement).max())
    assert csv_error < 1e-7, f"displacement.csv disagrees with structures by {csv_error}"

    strain = atomic_strain(reference, displacement)
    bonds = bond_strain(reference, displacement)
    triangles = vacancy_geometry(reference, displacement, vacancies)
    u_xy = np.linalg.norm(displacement[:, :2], axis=1)
    columns = np.column_stack([
        np.arange(len(reference)), reference.arrays["original_site_index"], reference.positions[:, :2],
        strain["second_neighbours"], strain["E_xx"], strain["E_yy"], strain["E_xy"],
        strain["areal_strain"], strain["principal_max"], strain["principal_min"],
        strain["max_shear_strain"], u_xy, saved["nearest_vacancy_distance_ref"],
    ])
    np.savetxt(
        directory / "strain.csv", columns, delimiter=",", fmt="%.12g", comments="",
        header="atom_index,original_site_index,x_ref,y_ref,second_neighbours,E_xx,E_yy,E_xy,areal_strain,principal_strain_max,principal_strain_min,max_shear_strain,u_xy_magnitude,nearest_vacancy_distance_ref",
    )
    np.savetxt(
        directory / "bond_strain.csv",
        np.column_stack([bonds[k] for k in ("i", "j", "x_mid_ref", "y_mid_ref", "reference_length", "relaxed_length", "strain")]),
        delimiter=",", fmt="%.12g", comments="",
        header="atom_i,atom_j,x_mid_ref,y_mid_ref,reference_length,relaxed_length,bond_strain",
    )
    (directory / "vacancy_geometry.json").write_text(json.dumps(triangles, indent=2) + "\n")
    lattice = json.loads((ROOT / "runs" / model["key"] / "lattice.json").read_text())
    triangle_distances = np.array([t["pair_distances_angstrom"] for t in triangles])
    return {
        "model_key": model["key"], "label": model["label"], "family": model["family"],
        "head": model["head"], "level_of_theory": model["level_of_theory"], "domain": model["domain"],
        "dtype": metrics["dtype"], "n": n, "carbon_atoms": metrics["carbon_atoms"],
        "steps": metrics["steps"], "bfgs_hessian_reset_steps": metrics.get("bfgs_hessian_reset_steps", []),
        "fmax_eV_per_angstrom": metrics["fmax_eV_per_angstrom"],
        "energy_change_from_ideal_eV": metrics["energy_change_from_ideal_eV"],
        "max_u_xy_angstrom": metrics["max_u_xy_angstrom"], "rms_u_xy_angstrom": metrics["rms_u_xy_angstrom"],
        "z_range_angstrom": metrics["z_range_angstrom"],
        "a_eq_angstrom": lattice["a_eq_angstrom"],
        "fixed_cell_biaxial_strain": lattice["fixed_cell_biaxial_strain"],
        "areal_strain_min": float(strain["areal_strain"].min()),
        "areal_strain_max": float(strain["areal_strain"].max()),
        "areal_strain_std": float(strain["areal_strain"].std()),
        "max_shear_strain_mean": float(strain["max_shear_strain"].mean()),
        "max_shear_strain_max": float(strain["max_shear_strain"].max()),
        "bond_strain_min": float(bonds["strain"].min()),
        "bond_strain_max": float(bonds["strain"].max()),
        "bond_strain_std": float(bonds["strain"].std()),
        "vacancy_pair_distance_min_angstrom": float(triangle_distances.min()),
        "vacancy_pair_distance_max_angstrom": float(triangle_distances.max()),
        "vacancy_reconstructed": bool(np.ptp(triangle_distances) > RECONSTRUCTION_SPREAD),
        "displacement_csv_max_difference_angstrom": csv_error,
    }


def compare_with_calculation_02(rows):
    """Compare the MACE-MPA-0 runs with the original calculation 02 results."""
    checks = []
    for n in SIZES:
        new = ROOT / "runs" / "mace-mpa-0-medium" / f"{n}x{n}"
        old = ROOT.parent / "calculations" / "02_vacancy_relaxation" / f"{n}x{n}"
        new_metrics = json.loads((new / "metrics.json").read_text())
        old_metrics = json.loads((old / "metrics.json").read_text())
        new_atoms, old_atoms = read(new / "relaxed.extxyz"), read(old / "relaxed.extxyz")
        reference_difference = np.abs(read(new / "reference.extxyz").positions - read(old / "reference.extxyz").positions).max()
        initial_difference = np.abs(read(new / "initial.extxyz").positions - read(old / "initial.extxyz").positions).max()
        checks.append({
            "n": n,
            "reference_max_position_difference_angstrom": float(reference_difference),
            "initial_max_position_difference_angstrom": float(initial_difference),
            "relaxed_max_position_difference_angstrom": float(np.abs(new_atoms.positions - old_atoms.positions).max()),
            "relaxed_energy_difference_eV": new_metrics["relaxed_energy_eV"] - old_metrics["relaxed_energy_eV"],
            "steps_new_old": [new_metrics["steps"], old_metrics["steps"]],
        })
    return checks


def label(row):
    return row["label"] + ("*" if row["domain"] != "materials" else "")


def markdown_tables(rows, models):
    order = {m["key"]: k for k, m in enumerate(models)}
    rows = sorted(rows, key=lambda r: (order[r["model_key"]], r["n"]))
    lines = [
        "| Model | Level of theory | Precision | $a_\\mathrm{eq}$ (Å) | Cell strain at $a = 2.46$ Å (%) |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        if row["n"] == SIZES[0]:
            lines.append(
                f"| {label(row)} | {row['level_of_theory'].replace('r2SCAN', 'r²SCAN').replace('wB97M', 'ωB97M')} | {row['dtype']} | "
                f"{row['a_eq_angstrom']:.4f} | {100 * row['fixed_cell_biaxial_strain']:+.2f} |"
            )
    lines += ["", "::: {.panel-tabset}", ""]
    for n in SIZES:
        lines += [
            f"### {n} × {n}", "",
            "| Model | Steps | Max $\\lvert\\mathbf{u}_\\parallel\\rvert$ (Å) | $\\varepsilon_A$ min (%) | $\\varepsilon_A$ max (%) "
            "| Max $\\gamma$ (%) | Bond strain min (%) | Bond strain max (%) | Vacancy-neighbour distances (Å) | $\\Delta E_\\mathrm{relax}$ (eV) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
        for row in (r for r in rows if r["n"] == n):
            lines.append(
                f"| {label(row)} | {row['steps']}{'†' if row['bfgs_hessian_reset_steps'] else ''} | {row['max_u_xy_angstrom']:.3f} | "
                f"{100 * row['areal_strain_min']:+.2f} | {100 * row['areal_strain_max']:+.2f} | "
                f"{100 * row['max_shear_strain_max']:.2f} | {100 * row['bond_strain_min']:+.2f} | "
                f"{100 * row['bond_strain_max']:+.2f} | "
                f"{row['vacancy_pair_distance_min_angstrom']:.3f}–{row['vacancy_pair_distance_max_angstrom']:.3f}"
                f"{', **reconstructed**' if row['vacancy_reconstructed'] else ''} | "
                f"{row['energy_change_from_ideal_eV']:.3f} |"
            )
        lines.append("")
    lines.append(":::")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    models = json.loads((ROOT / "models.json").read_text())["models"]
    tests = self_test()
    assert tests["zero_displacement_max_abs_strain"] < 1e-12
    assert tests["homogeneous_F_max_error"] < 1e-12
    assert tests["zero_displacement_max_abs_bond_strain"] < 1e-12
    rows = [analyse(model, n) for model in models for n in SIZES]
    with (ROOT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (ROOT / "_summary_tables.md").write_text(markdown_tables(rows, models))
    validation = {
        "strain_estimator_self_test": tests,
        "mace_mpa_0_vs_calculation_02": compare_with_calculation_02(rows),
        "all_runs_converged_and_fresh_evaluation_passed": True,
        "max_displacement_csv_difference_angstrom": max(r["displacement_csv_max_difference_angstrom"] for r in rows),
        "max_z_range_angstrom": max(r["z_range_angstrom"] for r in rows),
    }
    (ROOT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
