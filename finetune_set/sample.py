#!/usr/bin/env python3
"""Relax, and for some configurations run Langevin MD, with one generator model.

Each configuration in work/initial.extxyz names its generator. This script
processes the configurations assigned to one generator and writes one extended
XYZ file per configuration to work/sampled/<generator>/, containing the
starting structure, selected relaxation steps, the relaxed structure and the
saved MD frames. Finished configurations are skipped on restart.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from calculators import load_committee, make_calculator, prepare_environment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Process at most this many configurations (testing).")
    parser.add_argument("--reverse", action="store_true", help="Process configurations from the end (second worker).")
    parser.add_argument("--initial", type=Path, default=ROOT / "work" / "initial.extxyz")
    args = parser.parse_args()
    committee = load_committee()
    spec = next(m for m in committee["members"] if m["key"] == args.generator)
    device = args.device or committee["generators"][args.generator]["device"]
    prepare_environment(spec["kind"], device, args.threads)

    import numpy as np
    from ase import units
    from ase.io import read, write
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from ase.neighborlist import neighbor_list
    from ase.optimize import FIRE
    if spec["kind"] in ("mace", "mattersim", "upet", "deepmd"):
        import torch
        torch.set_num_threads(args.threads)

    config = json.loads((ROOT / "config.json").read_text())
    relax, md = config["relax"], config["md"]
    output = ROOT / "work" / "sampled" / args.generator
    output.mkdir(parents=True, exist_ok=True)
    todo = [a for a in read(args.initial, ":") if a.info["generator"] == args.generator]
    todo = [a for a in todo if not (output / f"{a.info['config_id']}.extxyz").exists()]
    if args.reverse:
        todo = todo[::-1]
    if args.limit:
        todo = todo[:args.limit]
    calc = make_calculator(spec, device=device, dtype=config["dtype"])
    log = (output / "progress.log").open("a")

    def snapshot(atoms, stage, step, **extra):
        frame = atoms.copy()
        frame.calc = None
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        frame.info.update(stage=stage, step=int(step), generator_energy=energy,
                          generator_fmax=float(np.linalg.norm(forces, axis=1).max()), **extra)
        return frame

    def healthy(atoms):
        if not np.all(np.isfinite(atoms.positions)):
            return False
        forces = atoms.get_forces()
        if not np.all(np.isfinite(forces)) or np.linalg.norm(forces, axis=1).max() > md["abort_max_force_eV_per_angstrom"]:
            return False
        _, _, d = neighbor_list("ijd", atoms, 0.7)
        return len(d) == 0

    for initial in todo:
        started = time.perf_counter()
        cid = initial.info["config_id"]
        if (output / f"{cid}.extxyz").exists():  # finished by another worker
            continue
        rng = np.random.default_rng(20261025 + int(cid[1:]))
        atoms = initial.copy()
        atoms.calc = calc
        frames = [snapshot(atoms, "initial", 0)]
        optimizer = FIRE(atoms, logfile=None)
        saved_steps = set(relax["saved_intermediate_steps"])

        def save_relax_step():
            if optimizer.nsteps in saved_steps:
                frames.append(snapshot(atoms, "relax", optimizer.nsteps))

        optimizer.attach(save_relax_step)
        status = "ok"
        try:
            converged = bool(optimizer.run(fmax=relax["fmax_eV_per_angstrom"], steps=relax["max_steps"]))
            frames.append(snapshot(atoms, "relaxed", optimizer.nsteps, relax_converged=converged))
        except Exception as error:  # an unphysical start can make a model fail; keep what was saved
            status = f"relax failed: {error}"
            converged = False
        run_md = status == "ok" and rng.random() < md["fraction_of_configs"]
        md_frames = 0
        if run_md:
            temperatures = md["temperatures_K"]
            T = float(rng.choice([float(t) for t in temperatures], p=np.array(list(temperatures.values())) / sum(temperatures.values())))
            dt = md["timestep_fs_with_hydrogen"] if 1 in atoms.numbers else md["timestep_fs"]
            MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=rng)
            Stationary(atoms)
            dyn = Langevin(atoms, timestep=dt * units.fs, temperature_K=T,
                           friction=md["friction_per_fs"] / units.fs, rng=rng)
            n_steps = int(round(md["total_time_fs"] / dt))
            every = int(round(md["save_every_fs"] / dt))
            first = int(round(md["equilibration_fs"] / dt))
            for block in range(n_steps // every):
                try:
                    dyn.run(every)
                except Exception as error:
                    status = f"md stopped: {error}"
                    break
                step = (block + 1) * every
                if not healthy(atoms):
                    status = f"md stopped at {step * dt:.0f} fs: unphysical"
                    break
                if step >= first:
                    frames.append(snapshot(atoms, "md", step, md_temperature_K=T, md_time_fs=step * dt, md_timestep_fs=dt))
                    md_frames += 1
        temporary = output / f"{cid}.extxyz.part"
        write(temporary, frames, format="extxyz")
        temporary.replace(output / f"{cid}.extxyz")
        log.write(json.dumps({"config_id": cid, "frames": len(frames), "relax_steps": optimizer.nsteps,
                              "relax_converged": converged, "md_frames": md_frames, "status": status,
                              "seconds": round(time.perf_counter() - started, 1)}) + "\n")
        log.flush()


if __name__ == "__main__":
    main()
