#!/usr/bin/env python3
"""Relax two-vacancy graphene cells and calculate unit-cell-referenced formation energies."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations, product


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import torch
from ase.build import graphene
from ase.geometry import find_mic
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import MACECalculator


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_force(atoms):
    return float(np.linalg.norm(atoms.get_forces(), axis=1).max())


def check_vacancy_geometry(config):
    """Repeat actual parent carbon sites 7x7 and check ordinary distances."""
    checks = []
    for n in range(2, 11):
        pristine = graphene(
            a=config["lattice_constant_angstrom"], size=(n, n, 1),
            vacuum=config["vacuum_per_side_angstrom"],
        )
        removed = config["vacancy_indices_zero_based"][str(n)]
        target = [2 / 3, 1 / 3] if n % 3 == 1 else [1 / 3, 2 / 3]
        np.testing.assert_allclose(
            pristine.get_scaled_positions()[removed, :2], [[0, 0], target], atol=1e-12,
        )
        tags = np.zeros(len(pristine), dtype=int)
        tags[removed] = 1
        pristine.set_tags(tags)
        repeated = pristine.repeat((7, 7, 1))
        sites = repeated.positions[repeated.get_tags() == 1]
        assert len(sites) == 98
        distance_spread, angle_error = [], []
        central_checks = []
        for i, j in product(range(1, 6), repeat=2):
            for index in removed:
                center = pristine.positions[index] + i * pristine.cell[0] + j * pristine.cell[1]
                vectors = sites - center
                distances = np.linalg.norm(vectors, axis=1)
                distances[distances < 1e-9] = np.inf
                order = np.argsort(distances)
                nearest = distances[order[:3]]
                unit = vectors[order[:3]] / nearest[:, None]
                angles = np.array([
                    np.degrees(np.arccos(np.clip(np.dot(unit[a], unit[b]), -1, 1)))
                    for a, b in combinations(range(3), 2)
                ])
                np.testing.assert_allclose(nearest, n * config["lattice_constant_angstrom"] / np.sqrt(3), atol=1e-9)
                np.testing.assert_allclose(angles, 120, atol=1e-9)
                assert distances[order[3]] > nearest[0] + 1e-9
                distance_spread.append(float(np.ptp(nearest)))
                angle_error.append(float(np.max(np.abs(angles - 120))))
                if (i, j) == (3, 3):
                    central_checks.append({
                        "original_site_index": index,
                        "nearest_displacements_angstrom": vectors[order[:3]].tolist(),
                        "nearest_distances_angstrom": nearest.tolist(),
                        "pairwise_angles_degrees": angles.tolist(),
                    })
        checks.append({
            "n": n, "vacancy_indices_zero_based": removed,
            "vacancy_fractional_positions": [[0, 0], target],
            "parent_sublattices": ["A", "A" if removed[1] % 2 == 0 else "B"],
            "nearest_vacancy_spacing_angstrom": n * config["lattice_constant_angstrom"] / np.sqrt(3),
            "explicit_repeat": [7, 7, 1], "explicit_vacancy_sites": len(sites),
            "interior_sites_checked": len(distance_spread),
            "maximum_distance_spread_angstrom": max(distance_spread),
            "maximum_angle_error_degrees": max(angle_error),
            "central_checks": central_checks,
        })
    (ROOT / "vacancy_geometry_check.json").write_text(json.dumps(checks, indent=2) + "\n")
    return {item["n"]: item for item in checks}


def perfect_reference(config, calculator, model_hash):
    """Use the two-carbon, unstrained unit cell at the same fixed a as defects."""
    output = ROOT / "perfect_unitcell"
    output.mkdir(exist_ok=True)
    atoms = graphene(
        a=config["lattice_constant_angstrom"],
        vacuum=config["vacuum_per_side_angstrom"],
    )
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    fmax = max_force(atoms)
    assert len(atoms) == 2 and np.isfinite(energy)
    assert fmax < config["fmax_eV_per_angstrom"], "Perfect unit cell has nonzero internal forces"
    chemical_potential = energy / len(atoms)
    atoms.info["carbon_chemical_potential_eV"] = chemical_potential
    write(output / "reference.extxyz", atoms, format="extxyz")
    record = {
        "carbon_atoms": len(atoms), "energy_eV": energy,
        "carbon_chemical_potential_eV": chemical_potential,
        "fmax_eV_per_angstrom": fmax,
        "lattice_constant_angstrom": config["lattice_constant_angstrom"],
        "vacuum_per_side_angstrom": config["vacuum_per_side_angstrom"],
        "reference_policy": "Single point on ideal two-carbon graphene unit cell; fixed lattice matching all defect cells.",
        "model_name": config["model_name"], "model_sha256": model_hash,
        "computed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Perfect graphene: E_unit={energy:.10f} eV; mu_C={chemical_potential:.10f} eV/C", flush=True)
    return record


def extract_formation(n, config, calculator, reference, geometry):
    """Verify saved energies and extensivity, then apply E_def - N_C mu_C."""
    output = ROOT / f"{n}x{n}"
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["model_sha256"] == reference["model_sha256"]
    pristine = graphene(
        a=config["lattice_constant_angstrom"], size=(n, n, 1),
        vacuum=config["vacuum_per_side_angstrom"],
    )
    pristine.calc = calculator
    perfect_energy = float(pristine.get_potential_energy())
    mu = reference["carbon_chemical_potential_eV"]
    extensivity_error = perfect_energy - len(pristine) * mu
    np.testing.assert_allclose(extensivity_error, 0, atol=1e-7, rtol=0)
    atoms = read(output / "relaxed.extxyz")
    saved_energy = atoms.get_potential_energy()
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    force = max_force(atoms)
    np.testing.assert_allclose(energy, metrics["relaxed_energy_eV"], atol=2e-7, rtol=0)
    np.testing.assert_allclose(energy, saved_energy, atol=2e-7, rtol=0)
    assert len(atoms) == 2 * n**2 - 2
    np.testing.assert_allclose(atoms.cell, pristine.cell, atol=1e-8)
    np.testing.assert_array_equal(atoms.pbc, [True, True, False])
    if metrics["converged"]:
        assert force <= config["fmax_eV_per_angstrom"] + 1e-7
    formation = energy - len(atoms) * mu
    np.testing.assert_allclose(formation, energy - perfect_energy + 2 * mu, atol=1e-7, rtol=0)
    row = {
        "n": n, "carbon_atoms": len(atoms), "vacancies": 2,
        "vacancy_fraction": 2 / (2 * n**2),
        "parent_sublattices": "/".join(geometry["parent_sublattices"]),
        "vacancy_spacing_angstrom": geometry["nearest_vacancy_spacing_angstrom"],
        "converged": metrics["converged"], "bfgs_steps": metrics["steps"],
        "fmax_eV_per_angstrom": force,
        "defect_energy_eV": energy, "mu_C_eV_per_atom": mu,
        "perfect_supercell_energy_eV": perfect_energy,
        "perfect_extensivity_error_eV": extensivity_error,
        "formation_energy_two_vacancies_eV": formation,
        "formation_energy_per_vacancy_eV": formation / 2,
        "ideal_defect_formation_energy_eV": metrics["ideal_defective_energy_eV"] - len(atoms) * mu,
        "relaxation_energy_eV": metrics["energy_change_from_ideal_eV"],
    }
    (output / "formation_energy.json").write_text(json.dumps(row, indent=2) + "\n")
    print(f"{n}x{n}: E_form(two vacancies)={formation:.8f} eV; per vacancy={formation / 2:.8f} eV", flush=True)
    return row


def write_summary():
    rows = []
    for n in range(2, 11):
        path = ROOT / f"{n}x{n}" / "formation_energy.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    if rows:
        with (ROOT / "formation_energies.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (ROOT / "formation_energies.json").write_text(json.dumps(rows, indent=2) + "\n")


def relax_size(n, config, calculator, model_hash, resume=False):
    output = ROOT / f"{n}x{n}"
    output.mkdir(exist_ok=True)
    settings = dict(config, n=n, model_sha256=model_hash)
    settings_path = output / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise RuntimeError(f"Settings changed for {output}; use a new calculation directory.")
        metrics_path = output / "metrics.json"
        if metrics_path.exists() and json.loads(metrics_path.read_text())["converged"]:
            print(f"{n}x{n}: already converged; keeping existing result", flush=True)
            return json.loads(metrics_path.read_text())
        if not resume:
            raise RuntimeError(f"Existing unfinished run in {output}; use --resume.")
    else:
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    pristine = graphene(
        a=config["lattice_constant_angstrom"], size=(n, n, 1),
        vacuum=config["vacuum_per_side_angstrom"],
    )
    removed = np.array(config["vacancy_indices_zero_based"][str(n)], dtype=int)
    keep = np.ones(len(pristine), dtype=bool)
    keep[removed] = False
    reference = pristine[keep]
    reference.set_array("original_site_index", np.flatnonzero(keep))
    reference.info.update(n=n, vacancy_indices=removed.tolist())
    reference.calc = calculator
    ideal_energy = reference.get_potential_energy()
    write(output / "reference.extxyz", reference, format="extxyz")
    vacancies = pristine[removed]
    vacancies.set_chemical_symbols(["X", "X"])
    write(output / "vacancy_sites.extxyz", vacancies, format="extxyz")

    trajectory = output / "relax.traj"
    restart = output / "optimizer.json"
    previous_steps = 0
    if resume and trajectory.exists():
        frames = read(trajectory, index=":")
        atoms = frames[-1]
        previous_steps = len(frames) - 1
        perturbed_energy = read(output / "initial.extxyz").get_potential_energy()
    else:
        atoms = reference.copy()
        rng = np.random.default_rng(config["seed"] + n)
        perturbation = rng.normal(
            scale=config["initial_perturbation_std_angstrom"], size=(len(atoms), 3)
        )
        perturbation -= perturbation.mean(axis=0)
        atoms.positions += perturbation
        atoms.calc = calculator
        perturbed_energy = atoms.get_potential_energy()
        write(output / "initial.extxyz", atoms, format="extxyz")
    atoms.calc = calculator
    optimizer = BFGS(
        atoms, restart=str(restart), trajectory=str(trajectory),
        logfile=str(output / "relax.log"),
        maxstep=config["max_step_angstrom"], append_trajectory=resume,
    )
    started = time.perf_counter()

    def report():
        print(
            f"{n}x{n}: step {previous_steps + optimizer.nsteps}, "
            f"E={atoms.get_potential_energy():.8f} eV, "
            f"fmax={max_force(atoms):.6f} eV/Angstrom", flush=True,
        )

    optimizer.attach(report, interval=25)
    converged = bool(optimizer.run(
        fmax=config["fmax_eV_per_angstrom"], steps=config["max_steps"]
    ))
    final_energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(final_energy) and np.all(np.isfinite(forces))
    np.testing.assert_allclose(atoms.cell, reference.cell, atol=1e-12)
    np.testing.assert_array_equal(atoms.pbc, [True, True, False])

    displacement, _ = find_mic(
        atoms.positions - reference.positions, reference.cell, reference.pbc
    )
    drift = displacement.mean(axis=0)
    displacement -= drift
    magnitude_xy = np.linalg.norm(displacement[:, :2], axis=1)
    magnitude_3d = np.linalg.norm(displacement, axis=1)
    relative_vacancy_vectors = reference.positions[:, None, :] - vacancies.positions[None, :, :]
    _, vacancy_distances = find_mic(
        relative_vacancy_vectors.reshape(-1, 3), reference.cell, reference.pbc
    )
    vacancy_distance = vacancy_distances.reshape(len(reference), 2).min(axis=1)
    atoms.set_array("u_xyz", displacement)
    atoms.set_array("u_xy_magnitude", magnitude_xy)
    write(output / "relaxed.extxyz", atoms, format="extxyz")
    columns = np.column_stack([
        np.arange(len(reference)), reference.arrays["original_site_index"],
        reference.positions, atoms.positions, displacement,
        magnitude_xy, magnitude_3d, vacancy_distance,
    ])
    np.savetxt(
        output / "displacement.csv", columns, delimiter=",", fmt="%.12g",
        header="atom_index,original_site_index,x_ref,y_ref,z_ref,x_relaxed,y_relaxed,z_relaxed,ux,uy,uz,u_xy_magnitude,u_3d_magnitude,nearest_vacancy_distance_ref",
        comments="",
    )
    metrics = {
        "n": n, "carbon_atoms": len(atoms), "converged": converged,
        "optimizer": "ASE BFGS", "steps": previous_steps + optimizer.nsteps,
        "fmax_eV_per_angstrom": float(np.linalg.norm(forces, axis=1).max()),
        "force_tolerance_eV_per_angstrom": config["fmax_eV_per_angstrom"],
        "ideal_defective_energy_eV": float(ideal_energy),
        "perturbed_initial_energy_eV": float(perturbed_energy),
        "relaxed_energy_eV": float(final_energy),
        "energy_change_from_ideal_eV": float(final_energy - ideal_energy),
        "max_u_xy_angstrom": float(magnitude_xy.max()),
        "rms_u_xy_angstrom": float(np.sqrt(np.mean(magnitude_xy**2))),
        "max_u_3d_angstrom": float(magnitude_3d.max()),
        "z_range_angstrom": float(np.ptp(atoms.positions[:, 2])),
        "removed_rigid_translation_angstrom": drift.tolist(),
        "cell_policy": "fixed; all atomic coordinates free in 3D",
        "displacement_definition": "Minimum-image final minus ideal surviving-site positions, minus mean rigid translation; primary plotted magnitude sqrt(ux^2+uy^2).",
        "strain_tensor": "Not computed; displacement magnitude is not strain.",
        "model_name": config["model_name"], "model_sha256": model_hash,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, choices=range(2, 11), default=list(range(2, 11)))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    assert config["cell_policy"] == "fixed"
    geometry = check_vacancy_geometry(config)
    model = (ROOT / config["model_path"]).resolve()
    model_hash = sha256(model)
    provenance = json.loads((ROOT / "model_provenance.json").read_text())
    assert model_hash == provenance["sha256"], "Checkpoint checksum changed"
    torch.set_num_threads(config["cpu_threads"])
    environment = {
        "python_executable": sys.executable, "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("mace-torch", "torch", "ase", "numpy", "e3nn", "scipy", "matplotlib")},
        "device": config["device"], "dtype": config["dtype"],
        "cpu_threads": config["cpu_threads"],
    }
    (ROOT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    calculator = MACECalculator(
        model_paths=str(model), device=config["device"], default_dtype=config["dtype"]
    )
    assert 6 in calculator.models[0].atomic_numbers.tolist()
    reference = perfect_reference(config, calculator, model_hash)
    results = []
    for n in args.sizes:
        results.append(relax_size(n, config, calculator, model_hash, args.resume))
        extract_formation(n, config, calculator, reference, geometry[n])
        write_summary()
    if not all(result["converged"] for result in results):
        raise SystemExit("At least one relaxation has not reached the force tolerance.")


if __name__ == "__main__":
    main()
