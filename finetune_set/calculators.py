"""ASE calculators for the committee members, imported lazily so each conda environment needs only its own package."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def load_committee():
    return json.loads((ROOT / "committee.json").read_text())


def member(key):
    for m in load_committee()["members"]:
        if m["key"] == key:
            return m
    raise KeyError(key)


def make_calculator(spec, device="cpu", dtype="float32"):
    path = PROJECT / spec["path"]
    kind = spec["kind"]
    if kind == "mace":
        from mace.calculators import MACECalculator
        head = {"head": spec["head"]} if spec["head"] else {}
        calc = MACECalculator(model_paths=str(path), device=device, default_dtype=dtype, **head)
        if spec["head"] and calc.head != spec["head"]:
            raise RuntimeError(f"Head {spec['head']} unavailable: {calc.available_heads}")
        return calc
    if kind == "mattersim":
        from mattersim.forcefield.potential import MatterSimCalculator, Potential
        potential = Potential.from_checkpoint(load_path=str(path), device=device, load_training_state=False)
        return MatterSimCalculator(potential=potential, device=device)
    if kind == "upet":
        import torch
        from upet.calculator import UPETCalculator
        return UPETCalculator(model=spec["upet_model"], version=spec["upet_version"], checkpoint_path=str(path),
                              dtype=getattr(torch, dtype), device=device)
    if kind == "deepmd":
        if device == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from deepmd.calculator import DP
        return DP(model=str(path))
    if kind == "grace":
        if device == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from tensorpotential.calculator import TPCalculator
        return TPCalculator(model=str(path))
    raise ValueError(kind)


def prepare_environment(kind, device, threads):
    """Thread and GPU-memory settings that must be in place before the model package is imported."""
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("DP_INTRA_OP_PARALLELISM_THREADS", str(threads))
    os.environ.setdefault("DP_INTER_OP_PARALLELISM_THREADS", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
