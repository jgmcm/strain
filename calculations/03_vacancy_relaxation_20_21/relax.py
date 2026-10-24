#!/usr/bin/env python3
"""Relax two-vacancy graphene cells and export atom-resolved displacement."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


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
    parser.add_argument("--sizes", nargs="+", type=int, choices=(20, 21), default=[20, 21])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    assert config["cell_policy"] == "fixed"
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
    results = [relax_size(n, config, calculator, model_hash, args.resume) for n in args.sizes]
    if not all(result["converged"] for result in results):
        raise SystemExit("At least one relaxation has not reached the force tolerance.")


if __name__ == "__main__":
    main()
