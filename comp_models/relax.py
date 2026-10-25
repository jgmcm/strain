#!/usr/bin/env python3
"""Relax the two-vacancy 4 x 4 and 5 x 5 graphene cells with each registered model."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
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
from scipy.optimize import minimize_scalar


PACKAGES = {
    "mace": ("mace-torch", "torch", "ase", "numpy", "scipy", "e3nn", "cuequivariance-torch"),
    "mattersim": ("mattersim", "torch", "ase", "numpy", "scipy", "e3nn"),
}


def current_env():
    found = [name for name in PACKAGES if importlib.util.find_spec(name)]
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one of {list(PACKAGES)} to be importable, found {found}")
    return found[0]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_force(atoms):
    return float(np.linalg.norm(atoms.get_forces(), axis=1).max())


def model_dtype(model, config):
    return config["dtype_overrides"].get(model["env"], config["dtype"])


def make_calculator(model, path, config):
    dtype = model_dtype(model, config)
    if model["env"] == "mace":
        from mace.calculators import MACECalculator
        head = {"head": model["head"]} if model["head"] else {}
        calculator = MACECalculator(
            model_paths=str(path), device=config["device"], default_dtype=dtype, **head
        )
        if model["head"] and calculator.head != model["head"]:
            raise RuntimeError(f"Head {model['head']} unavailable; heads are {calculator.available_heads}")
        network = calculator.models[0]
        assert 6 in network.atomic_numbers.tolist()
        details = {"head": calculator.head, "available_heads": list(calculator.available_heads)}
    else:
        from mattersim.forcefield.potential import MatterSimCalculator, Potential
        potential = Potential.from_checkpoint(
            load_path=str(path), device=config["device"], load_training_state=False
        )
        state = torch.load(path, map_location=config["device"])["model"]
        missing, unexpected = potential.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint keys do not match: {missing=}, {unexpected=}")
        calculator = MatterSimCalculator(potential=potential, device=config["device"])
        network = potential.model
        details = {"head": None, "available_heads": None}
    dtypes = {str(p.dtype) for p in network.parameters()}
    details.update(
        parameters=int(sum(p.numel() for p in network.parameters())),
        parameter_dtypes=sorted(dtypes), dtype=dtype,
    )
    if dtypes != {f"torch.{dtype}"}:
        raise RuntimeError(f"Model parameters are {dtypes}, expected {dtype}")
    return calculator, details


def equilibrium_lattice(config, calculator):
    def energy(a):
        cell = graphene(a=a, size=(1, 1, 1), vacuum=config["vacuum_per_side_angstrom"])
        cell.calc = calculator
        return cell.get_potential_energy() / len(cell)

    lower, upper = config["lattice_scan_bounds_angstrom"]
    result = minimize_scalar(
        energy, bounds=(lower, upper), method="bounded",
        options={"xatol": config["lattice_scan_tolerance_angstrom"]},
    )
    a_eq = float(result.x)
    if not result.success or min(a_eq - lower, upper - a_eq) < 1e-3:
        raise RuntimeError(f"Lattice scan did not find an interior minimum: {result}")
    a_fixed = config["lattice_constant_angstrom"]
    return {
        "a_eq_angstrom": a_eq,
        "energy_per_atom_at_a_eq_eV": float(result.fun),
        "energy_per_atom_at_fixed_a_eV": float(energy(a_fixed)),
        "fixed_a_angstrom": a_fixed,
        "fixed_cell_biaxial_strain": a_fixed / a_eq - 1,
        "method": "Bounded scalar minimisation of the perfect two-atom primitive-cell energy per atom over a; ideal positions, same vacuum.",
    }


def relax_size(n, model, config, calculator, model_hash, resume=False, reset_hessian=False):
    output = ROOT / "runs" / model["key"] / f"{n}x{n}"
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(config, n=n, model=model, model_sha256=model_hash)
    settings_path = output / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise RuntimeError(f"Settings changed for {output}; use a new calculation directory.")
        metrics_path = output / "metrics.json"
        if metrics_path.exists() and json.loads(metrics_path.read_text())["converged"]:
            print(f"{model['key']} {n}x{n}: already converged; keeping existing result", flush=True)
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
    previous_metrics = output / "metrics.json"
    hessian_resets = json.loads(previous_metrics.read_text()).get("bfgs_hessian_reset_steps", []) if resume and previous_metrics.exists() else []
    if resume and trajectory.exists():
        frames = read(trajectory, index=":")
        atoms = frames[-1]
        atoms.set_array("original_site_index", reference.arrays["original_site_index"])
        atoms.info.update(reference.info)
        previous_steps = len(frames) - 1
        perturbed_energy = read(output / "initial.extxyz").get_potential_energy()
        if reset_hessian:
            restart.unlink(missing_ok=True)
            hessian_resets.append(previous_steps)
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
            f"{model['key']} {n}x{n}: step {previous_steps + optimizer.nsteps}, "
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

    fresh = read(output / "relaxed.extxyz")
    fresh.calc = calculator
    fresh_forces = fresh.get_forces()
    fresh_check = {
        "fresh_fmax_eV_per_angstrom": float(np.linalg.norm(fresh_forces, axis=1).max()),
        "energy_difference_from_saved_eV": float(fresh.get_potential_energy() - final_energy),
        "max_force_difference_from_saved_eV_per_angstrom": float(np.abs(fresh_forces - forces).max()),
    }
    fresh_check["energy_tolerance_eV"] = 1e-6 if model_dtype(model, config) == "float64" else 1e-4
    fresh_check["passed"] = bool(
        fresh_check["fresh_fmax_eV_per_angstrom"] < config["fmax_eV_per_angstrom"]
        and abs(fresh_check["energy_difference_from_saved_eV"]) < fresh_check["energy_tolerance_eV"]
    )
    metrics = {
        "n": n, "carbon_atoms": len(atoms), "converged": converged,
        "optimizer": "ASE BFGS", "steps": previous_steps + optimizer.nsteps,
        "bfgs_hessian_reset_steps": hessian_resets,
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
        "fresh_evaluation": fresh_check,
        "model_key": model["key"], "model_label": model["label"], "head": model["head"],
        "model_sha256": model_hash, "dtype": model_dtype(model, config),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"{model['key']} {n}x{n}: converged={converged}, steps={metrics['steps']}, "
        f"fmax={metrics['fmax_eV_per_angstrom']:.6f}, max|u_xy|={metrics['max_u_xy_angstrom']:.6f} A, "
        f"fresh check {'passed' if fresh_check['passed'] else 'FAILED'}", flush=True,
    )
    return metrics


def run_model(model, config, checkpoints, provenance, sizes, resume, reset_hessian):
    checkpoint = checkpoints[model["checkpoint"]]
    path = ROOT / ".models" / checkpoint["file"]
    model_hash = sha256(path)
    if model_hash != provenance["checkpoints"][model["checkpoint"]]["sha256"]:
        raise RuntimeError(f"Checkpoint checksum changed for {model['checkpoint']}")
    calculator, details = make_calculator(model, path, config)
    directory = ROOT / "runs" / model["key"]
    directory.mkdir(parents=True, exist_ok=True)
    lattice_path = directory / "lattice.json"
    if not lattice_path.exists():
        lattice = equilibrium_lattice(config, calculator)
        lattice.update(model_key=model["key"], model_sha256=model_hash, **details)
        lattice_path.write_text(json.dumps(lattice, indent=2) + "\n")
    results = [relax_size(n, model, config, calculator, model_hash, resume, reset_hessian) for n in sizes]
    return all(result["converged"] and result["fresh_evaluation"]["passed"] for result in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="Model keys from models.json (default: all for this environment).")
    parser.add_argument("--sizes", nargs="+", type=int, choices=(4, 5), default=[4, 5])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-hessian", action="store_true",
                        help="With --resume, continue from the last frame with a fresh BFGS Hessian.")
    args = parser.parse_args()
    if args.reset_hessian and not args.resume:
        parser.error("--reset-hessian requires --resume")
    config = json.loads((ROOT / "config.json").read_text())
    assert config["cell_policy"] == "fixed" and not config["dispersion_correction"]
    registry = json.loads((ROOT / "models.json").read_text())
    checkpoints = {c["id"]: c for c in registry["checkpoints"]}
    provenance = json.loads((ROOT / "model_provenance.json").read_text())
    env = current_env()
    models = [m for m in registry["models"] if m["env"] == env]
    if args.models:
        unknown = set(args.models) - {m["key"] for m in models}
        if unknown:
            raise SystemExit(f"Not registered for the {env} environment: {sorted(unknown)}")
        models = [m for m in models if m["key"] in args.models]
    torch.set_num_threads(config["cpu_threads"])
    environment = {
        "python_executable": sys.executable, "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES[env]},
        "device": config["device"],
        "dtype": config["dtype_overrides"].get(env, config["dtype"]),
        "cpu_threads": config["cpu_threads"],
    }
    (ROOT / f"environment_{env}.json").write_text(json.dumps(environment, indent=2) + "\n")
    failed = [m["key"] for m in models
              if not run_model(m, config, checkpoints, provenance, args.sizes, args.resume, args.reset_hessian)]
    if failed:
        raise SystemExit(f"Not converged or failed the fresh evaluation: {failed}")


if __name__ == "__main__":
    main()
