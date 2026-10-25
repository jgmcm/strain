#!/usr/bin/env python3
"""Evaluate committee members on the candidate pool, in batches, one member at a time.

Results go to work/committee/<member>.npz: energies (eV) and forces
(eV/Angstrom) for all atoms in pool order, plus the evaluation time.
MACE-MPA-0 also stores a 64-dimensional random projection of its per-atom
invariant descriptors, used later to measure structural diversity.

Batches hold up to `--max-atoms` atoms; a batch that runs out of GPU memory is
split in half and retried. The GPU is throttled: after each batch the script
sleeps for `--duty-pause` times the batch's run time, and the process may use at
most half of the GPU memory.
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from calculators import PROJECT, load_committee, make_calculator, prepare_environment  # noqa: E402

DESCRIPTOR_MEMBER = "mace-mpa-0-medium"
DESCRIPTOR_DIMENSIONS = 64


def batches(frames, order, max_atoms):
    batch, atoms = [], 0
    for index in order:
        if batch and atoms + len(frames[index]) > max_atoms:
            yield batch
            batch, atoms = [], 0
        batch.append(index)
        atoms += len(frames[index])
    if batch:
        yield batch


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def out_of_memory(error):
    return "out of memory" in str(error).lower() or type(error).__name__ == "OutOfMemoryError"


def run(frames, evaluate_batch, max_atoms, pause, order=None):
    """Evaluate all frames; returns per-frame energies and force arrays in pool order."""
    import numpy as np
    energies, forces = [None] * len(frames), [None] * len(frames)
    extras = {}
    queue = list(batches(frames, order or range(len(frames)), max_atoms))
    while queue:
        indices = queue.pop(0)
        started = time.perf_counter()
        try:
            result = evaluate_batch(indices)
        except Exception as error:
            if not out_of_memory(error) or len(indices) == 1:
                raise
            import torch
            torch.cuda.empty_cache()
            half = len(indices) // 2
            queue[:0] = [indices[:half], indices[half:]]
            continue
        for n, k in enumerate(indices):
            energies[k] = result["energy"][n]
            forces[k] = result["forces"][n]
            for key, values in result.get("extras", {}).items():
                extras.setdefault(key, [None] * len(frames))[k] = values[n]
        if pause > 0:
            time.sleep(pause * (time.perf_counter() - started))
    out = {"energy": np.array(energies, dtype=float), "forces": np.concatenate(forces).astype(np.float32)}
    for key, values in extras.items():
        out[key] = np.concatenate(values)
    return out


def split(array, natoms):
    import numpy as np
    return np.split(array, np.cumsum(natoms)[:-1])


def mace_evaluator(spec, frames, device, cueq=False):
    import numpy as np
    import torch
    from e3nn import o3
    from mace import data
    from mace.modules.utils import extract_invariant
    from mace.tools import torch_geometric, utils

    torch.set_default_dtype(torch.float32)
    model = torch.load(PROJECT / spec["path"], map_location=device, weights_only=False).float().to(device)
    if cueq:
        from mace.cli.convert_e3nn_cueq import run as e3nn_to_cueq
        model = e3nn_to_cueq(model, device=device).float().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    heads = getattr(model, "heads", None)
    head = spec["head"]
    if head is None:
        head = heads[0] if heads and len(heads) == 1 else "Default"
        if heads and head not in heads:
            head = next(h for h in heads if h.lower() == "default")
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cutoff = float(model.r_max)
    describe = spec["key"] == DESCRIPTOR_MEMBER
    irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
    n_invariant = irreps_out.dim // (irreps_out.lmax + 1) ** 2
    rng = np.random.default_rng(0)
    projection = torch.tensor(rng.normal(size=(int(model.num_interactions) * n_invariant, DESCRIPTOR_DIMENSIONS))
                              / np.sqrt(DESCRIPTOR_DIMENSIONS), dtype=torch.float32, device=device)

    def evaluate_batch(indices):
        dataset = [data.AtomicData.from_config(data.config_from_atoms(frames[k], head_name=head),
                                               z_table=z_table, cutoff=cutoff, heads=heads) for k in indices]
        batch = next(iter(torch_geometric.dataloader.DataLoader(dataset=dataset, batch_size=len(dataset)))).to(device)
        output = model(batch.to_dict(), compute_stress=False)
        natoms = [len(frames[k]) for k in indices]
        result = {"energy": output["energy"].detach().cpu().numpy(),
                  "forces": split(output["forces"].detach().cpu().numpy(), natoms)}
        if describe:
            feats = extract_invariant(output["node_feats"], num_layers=int(model.num_interactions),
                                      num_features=n_invariant, l_max=irreps_out.lmax)
            result["extras"] = {"descriptors": split((feats.detach() @ projection).cpu().numpy().astype(np.float16), natoms)}
        return result

    return evaluate_batch


def evaluate_mace_group(specs, frames, device, max_atoms, pause, order):
    """All MACE members on shared batches: each batch's graph is built once and passed through every model.

    The graph uses the largest cutoff (6 Angstrom). Models with a 5 Angstrom cutoff give zero
    weight to the longer edges through their radial cutoff function. Each model's head index
    is written into the batch before its forward pass.
    """
    import numpy as np
    import torch
    from e3nn import o3
    from mace import data
    from mace.modules.utils import extract_invariant
    from mace.tools import torch_geometric, utils

    torch.set_default_dtype(torch.float32)
    models, head_index = [], []
    for spec in specs:
        model = torch.load(PROJECT / spec["path"], map_location=device, weights_only=False).float().to(device)
        for parameter in model.parameters():
            parameter.requires_grad = False
        heads = getattr(model, "heads", None) or ["Default"]
        head = spec["head"] or (heads[0] if len(heads) == 1 else next(h for h in heads if h.lower() == "default"))
        models.append(model)
        head_index.append(heads.index(head))
    tables = {tuple(int(z) for z in m.atomic_numbers) for m in models}
    assert len(tables) == 1, "MACE members use different element tables"
    z_table = utils.AtomicNumberTable(list(next(iter(tables))))
    cutoff = max(float(m.r_max) for m in models)
    describer = next((k for k, s in enumerate(specs) if s["key"] == DESCRIPTOR_MEMBER), None)
    if describer is not None:
        irreps_out = o3.Irreps(str(models[describer].products[0].linear.irreps_out))
        n_invariant = irreps_out.dim // (irreps_out.lmax + 1) ** 2
        n_layers = int(models[describer].num_interactions)
        projection = torch.tensor(np.random.default_rng(0).normal(size=(n_layers * n_invariant, DESCRIPTOR_DIMENSIONS))
                                  / np.sqrt(DESCRIPTOR_DIMENSIONS), dtype=torch.float32, device=device)

    energies = [[None] * len(frames) for _ in specs]
    forces = [[None] * len(frames) for _ in specs]
    descriptors = [None] * len(frames)
    queue = list(batches(frames, order, max_atoms))
    timing = np.zeros(len(specs))
    total_atoms, done_atoms, clock = sum(len(f) for f in frames), 0, time.perf_counter()
    while queue:
        indices = queue.pop(0)
        started = time.perf_counter()
        natoms = [len(frames[k]) for k in indices]
        try:
            dataset = [data.AtomicData.from_config(data.config_from_atoms(frames[k]), z_table=z_table, cutoff=cutoff)
                       for k in indices]
            batch = next(iter(torch_geometric.dataloader.DataLoader(dataset=dataset, batch_size=len(dataset)))).to(device)
            base = batch.to_dict()
            results = []
            for n, model in enumerate(models):
                t0 = time.perf_counter()
                inputs = dict(base)
                inputs["head"] = torch.full_like(base["head"], head_index[n])
                inputs["positions"] = base["positions"].detach().clone()
                output = model(inputs, compute_stress=False)
                item = [output["energy"].detach().cpu().numpy(), split(output["forces"].detach().cpu().numpy(), natoms)]
                if n == describer:
                    feats = extract_invariant(output["node_feats"], num_layers=n_layers, num_features=n_invariant,
                                              l_max=irreps_out.lmax)
                    item.append(split((feats.detach() @ projection).cpu().numpy().astype(np.float16), natoms))
                results.append(item)
                timing[n] += time.perf_counter() - t0
        except Exception as error:
            if not out_of_memory(error) or len(indices) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(indices) // 2
            queue[:0] = [indices[:half], indices[half:]]
            continue
        done_before = done_atoms
        done_atoms += sum(natoms)
        if done_atoms * 20 // total_atoms > done_before * 20 // total_atoms:
            elapsed = time.perf_counter() - clock
            print(f"MACE group: {100 * done_atoms / total_atoms:.0f} % of atoms after {elapsed / 60:.1f} min, "
                  f"about {elapsed * (total_atoms / done_atoms - 1) / 60:.0f} min left", flush=True)
        for n, item in enumerate(results):
            for j, k in enumerate(indices):
                energies[n][k] = item[0][j]
                forces[n][k] = item[1][j]
                if n == describer:
                    descriptors[k] = item[2][j]
        if pause > 0:
            time.sleep(pause * (time.perf_counter() - started))
    out = {}
    for n, spec in enumerate(specs):
        out[spec["key"]] = {"energy": np.array(energies[n], dtype=float),
                            "forces": np.concatenate(forces[n]).astype(np.float32), "model_seconds": timing[n]}
        if n == describer:
            out[spec["key"]]["descriptors"] = np.concatenate(descriptors)
    return out


def mattersim_evaluator(spec, frames, device):
    import numpy as np
    from mattersim.datasets.utils.build import build_dataloader
    from mattersim.forcefield.potential import Potential

    potential = Potential.from_checkpoint(load_path=str(PROJECT / spec["path"]), device=device, load_training_state=False)
    args = potential.model.model_args

    def evaluate_batch(indices):
        loader = build_dataloader([frames[k] for k in indices], model_type="m3gnet", cutoff=args["cutoff"],
                                  threebody_cutoff=args["threebody_cutoff"], batch_size=len(indices), only_inference=True)
        energies, forces, _ = potential.predict_properties(loader, include_forces=True, include_stresses=False)
        return {"energy": np.array(energies), "forces": [np.asarray(f) for f in forces]}

    return evaluate_batch


def deepmd_evaluator(spec, frames, device):
    import numpy as np
    from deepmd.infer import DeepPot

    dp = DeepPot(str(PROJECT / spec["path"]))
    type_map = {symbol: k for k, symbol in enumerate(dp.get_type_map())}

    def evaluate_batch(indices):
        energies, forces = [None] * len(indices), [None] * len(indices)
        groups = {}
        for n, k in enumerate(indices):
            groups.setdefault(len(frames[k]), []).append(n)
        for natoms, members in groups.items():
            chosen = [indices[n] for n in members]
            coords = np.array([frames[k].positions.reshape(-1) for k in chosen])
            cells = np.array([frames[k].cell[:].reshape(-1) for k in chosen])
            types = np.array([[type_map[s] for s in frames[k].get_chemical_symbols()] for k in chosen])
            e, f, _ = dp.eval(coords, cells, types, atomic=False, mixed_type=True)
            for m, n in enumerate(members):
                energies[n] = float(np.ravel(e[m])[0])
                forces[n] = np.asarray(f[m]).reshape(natoms, 3)
        return {"energy": np.array(energies), "forces": forces}

    return evaluate_batch


def ase_evaluator(spec, frames, device):
    """One structure at a time through the ASE calculator."""
    import numpy as np
    if spec["kind"] == "grace":
        from tensorpotential.calculator import TPCalculator
        calc = TPCalculator(model=str(PROJECT / spec["path"]), pad_neighbors_fraction=0.3, pad_atoms_number=8)
    else:
        calc = make_calculator(spec, device=device, dtype="float32")

    def evaluate_batch(indices):
        energies, forces = [], []
        for k in indices:
            atoms = frames[k].copy()
            atoms.calc = calc
            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())
        return {"energy": np.array(energies), "forces": forces}

    return evaluate_batch


EVALUATORS = {"mace": mace_evaluator, "mattersim": mattersim_evaluator, "deepmd": deepmd_evaluator,
              "upet": ase_evaluator, "grace": ase_evaluator}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, help="Evaluate all members that run in this conda environment.")
    parser.add_argument("--members", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-atoms", type=int, default=2000, help="Atoms per batch.")
    parser.add_argument("--duty-pause", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cueq", action="store_true", help="Convert MACE models to cuEquivariance kernels.")
    parser.add_argument("--no-group-mace", dest="group_mace", action="store_false",
                        help="Evaluate MACE members one by one instead of on shared batches.")
    parser.add_argument("--pool", type=Path, default=ROOT / "work" / "candidates.extxyz")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "committee")
    args = parser.parse_args()
    committee = load_committee()
    specs = [m for m in committee["members"] if m["env"] == args.env and (not args.members or m["key"] in args.members)]
    prepare_environment(specs[0]["kind"], args.device, args.threads)

    import numpy as np
    from ase.io import read
    if specs[0]["kind"] in ("mace", "mattersim", "upet", "deepmd"):
        import torch
        torch.set_num_threads(args.threads)
        if specs[0]["kind"] == "mace":
            # Without this, TorchScript re-profiles the MACE models for every new batch shape.
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        if args.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.5)
    frames = read(args.pool, ":")
    natoms = np.array([len(f) for f in frames])
    args.output.mkdir(parents=True, exist_ok=True)
    order = list(np.argsort(-natoms, kind="stable"))
    if specs[0]["kind"] == "mace" and args.group_mace:
        todo = [s for s in specs if not ((args.output / f"{s['key']}.npz").exists()
                                         and np.load(args.output / f"{s['key']}.npz")["energy"].shape[0] == len(frames))]
        if todo:
            started = time.perf_counter()
            results = evaluate_mace_group(todo, frames, args.device, args.max_atoms, args.duty_pause, order)
            elapsed = time.perf_counter() - started
            for spec in todo:
                result = results[spec["key"]]
                assert np.all(np.isfinite(result["energy"])) and np.all(np.isfinite(result["forces"])), "non-finite output"
                np.savez_compressed(args.output / f"{spec['key']}.npz", natoms=natoms, seconds=elapsed / len(todo),
                                    shared_batches=True, duty_pause=args.duty_pause, **result)
                print(f"{spec['key']}: model time {result['model_seconds']:.0f} s", flush=True)
            print(f"{len(todo)} MACE members: {len(frames)} frames in {elapsed:.0f} s", flush=True)
        return
    for spec in specs:
        target = args.output / f"{spec['key']}.npz"
        if target.exists() and np.load(target)["energy"].shape[0] == len(frames):
            print(f"{spec['key']}: already evaluated", flush=True)
            continue
        lock = target.with_suffix(".lock")
        if lock.exists() and process_alive(int(lock.read_text())):
            print(f"{spec['key']}: being evaluated by process {lock.read_text().strip()}; skipped", flush=True)
            continue
        lock.write_text(str(os.getpid()))
        started = time.perf_counter()
        # Largest structures first: GRACE then compiles one padded shape that covers the rest.
        evaluator = EVALUATORS[spec["kind"]](spec, frames, args.device, **({"cueq": True} if args.cueq and spec["kind"] == "mace" else {}))
        result = run(frames, evaluator, args.max_atoms, args.duty_pause, order)
        elapsed = time.perf_counter() - started
        assert result["energy"].shape == (len(frames),) and result["forces"].shape == (natoms.sum(), 3)
        assert np.all(np.isfinite(result["energy"])) and np.all(np.isfinite(result["forces"])), "non-finite output"
        np.savez_compressed(target, natoms=natoms, seconds=elapsed, duty_pause=args.duty_pause, **result)
        lock.unlink(missing_ok=True)
        print(f"{spec['key']}: {len(frames)} frames in {elapsed:.0f} s", flush=True)


if __name__ == "__main__":
    main()
