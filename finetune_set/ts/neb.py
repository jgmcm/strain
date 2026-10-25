#!/usr/bin/env python3
"""Relax the end states of every pair, then run climbing-image NEB, with one generator model, in batches.

All structures that need forces at an optimiser step (end states during the
relaxation, interior images during the NEB) are evaluated together, in batches
of up to `--max-atoms` atoms. Each structure or band keeps its own ASE
optimiser, which reads the batched result through a SinglePointCalculator.

The GPU is throttled as in the committee evaluation: after each batch the
script waits for `--duty-pause` times the batch's run time, less the time spent
building the next batch's graph on the CPU, and the process may use at most
half of the GPU memory.

Bands run in rounds. A band whose climbing image runs away uphill, or which
does not reach the climbing stage, is run again in the next round with about
twice as many images. A band that finds a state well below both end states is
split: its lowest image is relaxed, and the next round runs one band into that
intermediate minimum and one out of it. Follow-up bands get the parent's
pair_id with "r" (rerun), "a" (first half) or "b" (second half) appended.

Outputs in work/:
- endpoints.extxyz and endpoints.jsonl: relaxed end states and the checks on each pair;
  a restart reuses them;
- bands.extxyz and neb.jsonl: every image of each finished band and one record per band.
"""

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
from calculators import PROJECT, member, prepare_environment  # noqa: E402
from common import WORK, displacement, event_atoms, load_config, normal, unphysical  # noqa: E402
from evaluate import batches, out_of_memory  # noqa: E402


class Generator:
    """Batched energies and forces from one MACE model."""

    def __init__(self, spec, device, max_atoms, pause):
        import torch
        from mace import data
        from mace.tools import torch_geometric, utils
        # The TorchScript profiling executor re-optimises the model for every new batch shape,
        # which costs about 0.4 s per call when every batch differs.
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch.set_default_dtype(torch.float32)
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.5)
        self.torch, self.data, self.loader = torch, data, torch_geometric.dataloader.DataLoader
        self.model = torch.load(PROJECT / spec["path"], map_location=device, weights_only=False).float().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.heads = getattr(self.model, "heads", None)
        self.head = spec["head"] or (self.heads[0] if self.heads else "Default")
        self.z_table = utils.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        self.cutoff = float(self.model.r_max)
        self.device, self.max_atoms, self.pause = device, max_atoms, pause
        self.gpu_seconds, self.atoms_evaluated = 0.0, 0

    def graph(self, frames):
        dataset = [self.data.AtomicData.from_config(self.data.config_from_atoms(f, head_name=self.head),
                                                    z_table=self.z_table, cutoff=self.cutoff, heads=self.heads)
                   for f in frames]
        return next(iter(self.loader(dataset=dataset, batch_size=len(dataset)))).to(self.device)

    def __call__(self, frames):
        import numpy as np
        energies, forces = [None] * len(frames), [None] * len(frames)
        queue = list(batches(frames, range(len(frames)), self.max_atoms))
        prepared = None
        while queue:
            indices = queue.pop(0)
            batch = prepared if prepared is not None else self.graph([frames[k] for k in indices])
            prepared = None
            started = time.perf_counter()
            try:
                output = self.model(batch.to_dict(), compute_stress=False)
                e = output["energy"].detach().cpu().numpy()
                f = output["forces"].detach().cpu().numpy()
            except Exception as error:
                if not out_of_memory(error) or len(indices) == 1:
                    raise
                self.torch.cuda.empty_cache()
                half = len(indices) // 2
                queue[:0] = [indices[:half], indices[half:]]
                continue
            gpu = time.perf_counter() - started
            self.gpu_seconds += gpu
            natoms = [len(frames[k]) for k in indices]
            self.atoms_evaluated += sum(natoms)
            for n, (k, block) in enumerate(zip(indices, np.split(f, np.cumsum(natoms)[:-1]))):
                energies[k], forces[k] = float(e[n]), block
            started = time.perf_counter()
            if queue:  # build the next graph while the GPU rests
                prepared = self.graph([frames[k] for k in queue[0]])
            if self.pause > 0:
                time.sleep(max(0.0, self.pause * gpu - (time.perf_counter() - started)))
        return energies, forces


def attach(atoms, energy, forces):
    from ase.calculators.singlepoint import SinglePointCalculator
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)


def relax_all(states, generator, settings, label):
    """FIRE relaxation of all states at once; returns steps, final fmax and convergence per state."""
    import numpy as np
    from ase.optimize import FIRE
    optimizers = [FIRE(s, logfile=None) for s in states]
    steps = np.zeros(len(states), int)
    fmax = np.full(len(states), np.inf)
    active = list(range(len(states)))
    clock, iteration = time.perf_counter(), 0
    while active:
        energies, forces = generator([states[k] for k in active])
        still = []
        for n, k in enumerate(active):
            attach(states[k], energies[n], forces[n])
            fmax[k] = float(np.linalg.norm(forces[n], axis=1).max())
            if fmax[k] < settings["fmax_eV_per_angstrom"] or steps[k] >= settings["max_steps"]:
                continue
            optimizers[k].step()
            steps[k] += 1
            still.append(k)
        active = still
        iteration += 1
        if iteration % 25 == 0 or not active:
            print(f"{label}: step {iteration}, {len(active)} of {len(states)} still relaxing, "
                  f"{(time.perf_counter() - clock) / 60:.1f} min", flush=True)
    return steps, fmax, fmax < settings["fmax_eV_per_angstrom"]


def check_pair(initial, final, config, follow_up=False):
    """The reason a relaxed pair cannot be used for an NEB, or None.

    For a follow-up pair (one half of a split band) any atom may carry the change, so the
    displacement test uses all atoms with the lower threshold.
    """
    import numpy as np
    from ase.geometry import find_mic
    checks, relax = config["end_state_checks"], config["relax"]
    for state in (initial, final):
        if state.info["relax_fmax"] > relax["accept_fmax_eV_per_angstrom"]:
            return "relaxation not converged"
        if unphysical(state):
            return f"{state.info['state']} state {unphysical(state)}"
    d = np.linalg.norm(displacement(initial, final), axis=1)
    event = event_atoms(initial)
    small = checks["min_event_displacement_reconstruction_switch_angstrom"]
    if follow_up:
        if d.max() < small:
            return "end states relaxed to the same structure"
        return None
    threshold = small if initial.info["process"] == "V1 reconstruction switch" else checks["min_event_displacement_angstrom"]
    if d[event].max() < threshold:
        return "end states relaxed to the same structure"
    centre = initial.positions[event].mean(axis=0)
    _, r = find_mic(initial.positions - centre, initial.cell, initial.pbc)
    far = r > checks["far_field_radius_angstrom"]
    if far.any() and d[far].max() > checks["max_far_field_displacement_angstrom"]:
        return "end states differ away from the event"
    return None


def rigid_free_neb(images, rotation, **kwargs):
    """An ASE NEB whose image forces carry no net translation and, for tubes and cages, no net rotation.

    The NEB projections can leave a net force or torque on an image. A sheet can then slide within
    its plane, and a tube or cage in vacuum can also rotate, without any restoring force, and the
    band drifts apart. `rotation` is None (sheet), "axis" (tube, about z) or "all" (cage).
    """
    import numpy as np
    from ase.mep import NEB

    class RigidFreeNEB(NEB):
        def get_forces(self):
            forces = super().get_forces()
            n = self.natoms
            for m, image in enumerate(self.images[1:-1]):
                f = forces[m * n:(m + 1) * n]
                f -= f.mean(axis=0)
                if rotation is None:
                    continue
                r = image.positions - image.positions.mean(axis=0)
                if rotation == "axis":
                    r[:, 2] = 0.0
                    omega = np.array([0.0, 0.0, np.cross(r, f).sum(axis=0)[2] / (r ** 2).sum()])
                else:
                    inertia = (r ** 2).sum() * np.eye(3) - r.T @ r
                    omega = np.linalg.solve(inertia, np.cross(r, f).sum(axis=0))
                f -= np.cross(omega, r)
            return forces

    return RigidFreeNEB(images, **kwargs)


ROTATION = {"sheet": None, "nanotube": "axis", "fullerene": "all"}


def seed(band_id):
    return 20261025 + zlib.crc32(band_id.encode())


class Band:
    def __init__(self, band_id, initial, final, n_images, settings):
        import numpy as np
        from ase.optimize import FIRE
        rng = np.random.default_rng(seed(band_id))
        self.band_id, self.initial_info = band_id, dict(initial.info)
        images = [initial] + [initial.copy() for _ in range(n_images)] + [final]
        self.neb = rigid_free_neb(images, ROTATION[initial.info["framework"]], k=settings["spring_constant_min_eV_per_angstrom2"],
                                  climb=False, method=settings["method"])
        self.neb.interpolate(settings["interpolation"], mic=True)
        self.info = {}
        if initial.info["rotation"] and band_id[-1] not in "ab":  # not for the halves of a split band
            # Push the rotating pair out of the surface, so the band does not stay on the
            # in-plane rotation path by symmetry. Both atoms move to the same side, or to opposite sides.
            amplitude = rng.uniform(*settings["rotation_buckling_angstrom"])
            signs = np.array([1.0, rng.choice([-1.0, 1.0])]) * rng.choice([-1.0, 1.0])
            for m, image in enumerate(images[1:-1], start=1):
                for atom, sign in zip(event_atoms(initial), signs):
                    image.positions[atom] += sign * amplitude * np.sin(np.pi * m / (n_images + 1)) * normal(initial, image.positions[atom])
            self.info["rotation_buckling"] = float(amplitude * signs[0])
            self.info["rotation_buckling_same_side"] = int(signs[0] == signs[1])
        for image in images[1:-1]:
            image.positions += rng.normal(0, settings["interior_noise_angstrom"], image.positions.shape)
        self.optimizer = FIRE(self.neb, dt=settings["fire_dt"], maxstep=settings["fire_maxstep_angstrom"], logfile=None)
        self.settings, self.steps, self.status, self.fmax, self.climb_energy = settings, 0, None, None, None

    def interior(self):
        return self.neb.images[1:-1]

    def advance(self):
        """One optimiser step, after the interior forces are attached; False once the band is finished."""
        import numpy as np
        s = self.settings
        images = self.neb.images

        def fmax():
            return float(np.sqrt((self.neb.get_forces() ** 2).sum(axis=1)).max())

        energies = [image.get_potential_energy() for image in images]
        # Energy-weighted springs (Henkelman and Jonsson): stiffer springs between high-energy images
        # gather images near the top of the barrier, so a sharp barrier is resolved by more than one image.
        e = np.array(energies)
        reference, top = max(e[0], e[-1]), e[1:-1].max()
        weight = np.clip((np.maximum(e[:-1], e[1:]) - reference) / (top - reference), 0, 1) if top > reference else 0 * e[:-1]
        k_min, k_max = s["spring_constant_min_eV_per_angstrom2"], s["spring_constant_max_eV_per_angstrom2"]
        self.neb.k = list(k_min + (k_max - k_min) * weight)
        self.fmax = fmax()
        if not self.neb.climb and self.fmax < s["climb_when_fmax_below"]:
            # Climbing starts only near the path: an early climbing image can run far uphill.
            self.neb.climb = True
            self.optimizer.vel = None
            self.climb_energy = max(energies[1:-1])
            self.fmax = fmax()
        if self.neb.climb and self.fmax < s["fmax_eV_per_angstrom"]:
            self.status = "converged"
            return False
        if self.neb.climb:
            # Where the band resolves the barrier poorly, the climbing image can keep moving uphill
            # along a bad tangent and away from its neighbours.
            jump = max(np.linalg.norm(displacement(images[m], images[m + 1]), axis=1).max() for m in range(len(images) - 1))
            if max(energies[1:-1]) > self.climb_energy + s["max_climbing_rise_eV"] or jump > s["max_image_jump_angstrom"]:
                self.status = "climbing image ran away"
                return False
        if self.steps >= s["check_image_below_end_states_after_steps"] and \
                min(energies[1:-1]) < min(energies[0], energies[-1]) - s["stop_if_image_below_end_states_eV"]:
            self.status = "image below end states"
            return False
        if not self.neb.climb and self.steps >= s["give_up_without_climbing_after_steps"]:
            self.status = "no climbing stage"
            return False
        if self.steps >= s["max_steps"]:
            self.status = "converged (loose)" if self.neb.climb and self.fmax < s["accept_fmax_eV_per_angstrom"] else "not converged"
            return False
        self.optimizer.step()
        self.steps += 1
        return True

    def results(self, min_barrier):
        import numpy as np
        images = self.neb.images
        energies = np.array([image.get_potential_energy() for image in images])
        top = int(np.argmax(energies[1:-1])) + 1
        status = self.status
        if status in ("converged", "converged (loose)"):
            if energies[top] - max(energies[0], energies[-1]) < min_barrier:
                status = "no barrier"
            elif unphysical(images[top]):
                status = f"transition state {unphysical(images[top])}"
        # An interior image well below both neighbours marks an intermediate minimum on the path.
        inner = energies[1:-1]
        intermediate = any(inner[m] < min(energies[m], energies[m + 2]) - 0.05 for m in range(len(inner)))
        path = sum(np.linalg.norm(displacement(images[m], images[m + 1])) for m in range(len(images) - 1))
        return {"pair_id": self.band_id, "process": self.initial_info["process"],
                "generation": int(self.initial_info.get("generation", 0)), "parent": self.initial_info.get("parent", ""),
                "n_images": len(images) - 2, "status": status, "steps": self.steps, "fmax": self.fmax, "climbing_image": top,
                "barrier_forward": float(energies[top] - energies[0]), "barrier_backward": float(energies[top] - energies[-1]),
                "reaction_energy": float(energies[-1] - energies[0]), "intermediate_minimum": bool(intermediate),
                "path_length": float(path), "energies": [float(e) for e in energies], **self.info}


def write_band(band, record, handle):
    import numpy as np
    from ase.io import write
    frames = []
    for m, image in enumerate(band.neb.images):
        frame = image.copy()
        frame.calc = None
        forces = image.get_forces()
        for key in ("state", "relax_fmax", "relax_steps"):
            frame.info.pop(key, None)
        frame.info.update(pair_id=band.band_id, image=m, n_band_images=len(band.neb.images),
                          generator_energy=float(image.get_potential_energy()),
                          generator_fmax=float(np.linalg.norm(forces, axis=1).max()),
                          climbing=int(m == record["climbing_image"]), neb_status=record["status"],
                          neb_fmax=record["fmax"], barrier_forward=record["barrier_forward"],
                          barrier_backward=record["barrier_backward"], generation=record["generation"],
                          parent=record["parent"] or "none")
        frames.append(frame)
    write(handle, frames, format="extxyz")


def run_bands(jobs, generator, settings, label):
    """Run all bands together until each has finished; returns the finished Band objects."""
    bands = [Band(*job, settings) for job in jobs]
    print(f"{label}: {len(bands)} bands", flush=True)
    clock, iteration, active, finished = time.perf_counter(), 0, bands, []
    while active:
        images = [image for band in active for image in band.interior()]
        energies, forces = generator(images)
        position, still = 0, []
        for band in active:
            n = len(band.interior())
            for image, e, f in zip(band.interior(), energies[position:position + n], forces[position:position + n]):
                attach(image, e, f)
            position += n
            (still if band.advance() else finished).append(band)
        active = still
        iteration += 1
        if iteration % 20 == 0 or not active:
            rate = generator.atoms_evaluated / max(generator.gpu_seconds, 1e-9)
            print(f"{label} step {iteration}: {len(active)} bands running, {len(finished)} finished, "
                  f"{(time.perf_counter() - clock) / 60:.1f} min, GPU {generator.gpu_seconds / 60:.1f} min "
                  f"({rate:.0f} atoms/s)", flush=True)
    return finished


def follow_ups(finished, generator, config):
    """The next round's jobs: reruns with more images, and the two halves of split bands."""
    import numpy as np
    settings = config["neb"]
    jobs, splits, notes = [], [], {}
    for band in finished:
        generation = int(band.initial_info.get("generation", 0))
        if generation >= settings["max_generation"]:
            continue
        initial, final = band.neb.images[0], band.neb.images[-1]
        if band.status in ("climbing image ran away", "no climbing stage", "not converged") and not band.band_id.endswith("r"):
            n = min(2 * len(band.interior()) + 1, settings["max_images"])
            jobs.append((band.band_id + "r", child(initial, band, "r", generation), child(final, band, "r", generation), n))
            notes[band.band_id] = [band.band_id + "r"]
        elif band.status == "image below end states":
            energies = [image.get_potential_energy() for image in band.interior()]
            middle = band.interior()[int(np.argmin(energies))].copy()
            splits.append((band, middle))
    if splits:
        middles = [m for _, m in splits]
        steps, fmax, _ = relax_all(middles, generator, config["relax"], "intermediate minima")
        for (band, middle), n_steps, f in zip(splits, steps, fmax):
            generation = int(band.initial_info.get("generation", 0))
            initial, final = band.neb.images[0], band.neb.images[-1]
            middle.info.update(relax_steps=int(n_steps), relax_fmax=float(f))
            halves = []
            for suffix, a, b in (("a", initial, middle), ("b", middle, final)):
                first, second = child(a, band, suffix, generation), child(b, band, suffix, generation)
                first.info["state"], second.info["state"] = "initial", "final"
                if check_pair(first, second, config, follow_up=True) is None:
                    jobs.append((band.band_id + suffix, first, second, len(band.interior())))
                    halves.append(band.band_id + suffix)
            notes[band.band_id] = halves
    return jobs, notes


def child(state, band, suffix, generation):
    """A copy of an end state for a follow-up band, keeping its energy and forces."""
    atoms = state.copy()
    atoms.info.update(band.initial_info)
    atoms.info.update(pair_id=band.band_id + suffix, parent=band.band_id, generation=generation + 1,
                      relax_fmax=state.info.get("relax_fmax", 0.0))
    attach(atoms, state.get_potential_energy(), state.get_forces())
    return atoms


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, default=WORK / "pairs.extxyz")
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--max-atoms", type=int, default=2500)
    parser.add_argument("--duty-pause", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Use only the first pairs (testing).")
    parser.add_argument("--endpoints-only", action="store_true", help="Stop after relaxing the end states.")
    args = parser.parse_args()
    config = load_config()
    spec = member(config["generator"])
    prepare_environment(spec["kind"], config["device"], args.threads)

    import numpy as np
    import torch
    from ase.io import read, write
    torch.set_num_threads(args.threads)
    generator = Generator(spec, config["device"], args.max_atoms, args.duty_pause)
    work = args.work
    work.mkdir(exist_ok=True)
    started = time.perf_counter()

    endpoints = work / "endpoints.extxyz"
    if not endpoints.exists():
        states = read(args.pairs, ":")
        if args.limit:
            states = states[:2 * args.limit]
        steps, fmax, converged = relax_all(states, generator, config["relax"], "end states")
        for state, n, f in zip(states, steps, fmax):
            state.info.update(relax_steps=int(n), relax_fmax=float(f), generator_energy=float(state.get_potential_energy()))
        with (work / "endpoints.jsonl").open("w") as log:
            for k in range(0, len(states), 2):
                reason = check_pair(states[k], states[k + 1], config)
                log.write(json.dumps({"pair_id": states[k].info["pair_id"], "status": reason or "ok",
                                      "relax_steps": [int(steps[k]), int(steps[k + 1])],
                                      "relax_fmax": [float(fmax[k]), float(fmax[k + 1])]}) + "\n")
        for state in states:
            forces = state.get_forces()
            state.calc = None
            state.arrays["generator_forces"] = forces
        write(endpoints, states, format="extxyz")
        print(f"end states: {time.perf_counter() - started:.0f} s, GPU {generator.gpu_seconds:.0f} s", flush=True)

    if args.endpoints_only:
        return
    states = read(endpoints, ":")
    for state in states:
        attach(state, state.info["generator_energy"], state.arrays.pop("generator_forces"))
    status = {}
    for line in (work / "endpoints.jsonl").open():
        record = json.loads(line)
        status[record["pair_id"]] = record["status"]
    settings = config["neb"]
    jobs = [(states[k].info["pair_id"], states[k], states[k + 1], int(states[k].info["n_images"]))
            for k in range(0, len(states), 2) if status[states[k].info["pair_id"]] == "ok"]
    generation = 0
    with (work / "neb.jsonl").open("w") as log, (work / "bands.extxyz").open("w") as band_file:
        while jobs:
            finished = run_bands(jobs, generator, settings, f"NEB round {generation + 1}")
            jobs, notes = follow_ups(finished, generator, config)
            for band in finished:
                record = band.results(settings["min_barrier_eV"])
                record["follow_up"] = notes.get(band.band_id, [])
                log.write(json.dumps(record) + "\n")
                write_band(band, record, band_file)
            log.flush()
            band_file.flush()
            generation += 1
    print(f"total {time.perf_counter() - started:.0f} s, GPU {generator.gpu_seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()
