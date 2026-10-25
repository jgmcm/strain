#!/usr/bin/env python3
"""Copy or download every registered checkpoint into .models/ and verify it."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / ".models"


def digests(path):
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(block)
            git_blob.update(block)
    return sha256.hexdigest(), git_blob.hexdigest()


def fetch(checkpoint):
    target = MODELS / checkpoint["file"]
    source = "existing copy in .models/"
    if not target.exists():
        local = [Path(p).expanduser() for p in checkpoint.get("local_sources", [])]
        local = [p for p in local if p.is_file() and p.stat().st_size == checkpoint["bytes"]]
        partial = target.with_suffix(target.suffix + ".part")
        if local:
            shutil.copyfile(local[0], partial)
            source = f"copied from {local[0]}"
        else:
            print(f"Downloading {checkpoint['url']}", flush=True)
            with urllib.request.urlopen(checkpoint["url"], timeout=120) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            source = "downloaded from url"
        partial.replace(target)
    size = target.stat().st_size
    sha256, git_blob_sha1 = digests(target)
    checks = {"size_matches_upstream": size == checkpoint["bytes"]}
    if "expected_sha256" in checkpoint:
        checks["sha256_matches_upstream"] = sha256 == checkpoint["expected_sha256"]
    if "expected_git_blob_sha1" in checkpoint:
        checks["git_blob_sha1_matches_upstream"] = git_blob_sha1 == checkpoint["expected_git_blob_sha1"]
    if not all(checks.values()):
        raise RuntimeError(f"{target.name} failed verification: {checks}")
    print(f"{checkpoint['id']}: {target.name} ({size} bytes) {source}; checks passed", flush=True)
    return {
        "id": checkpoint["id"], "local_path": str(target.relative_to(ROOT)),
        "official_url": checkpoint["url"], "licence": checkpoint["licence"],
        "bytes": size, "sha256": sha256, "source": source, "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="Checkpoint ids to fetch (default: all).")
    args = parser.parse_args()
    registry = json.loads((ROOT / "models.json").read_text())
    checkpoints = [c for c in registry["checkpoints"] if not args.only or c["id"] in args.only]
    MODELS.mkdir(exist_ok=True)
    provenance_path = ROOT / "model_provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {"checkpoints": {}}
    for checkpoint in checkpoints:
        record = fetch(checkpoint)
        previous = provenance["checkpoints"].get(checkpoint["id"])
        if previous and previous["sha256"] != record["sha256"]:
            raise RuntimeError(f"{checkpoint['id']} checksum differs from model_provenance.json")
        if not previous:
            record["recorded_utc"] = datetime.now(timezone.utc).isoformat()
            provenance["checkpoints"][checkpoint["id"]] = record
    provenance["checkpoints"] = dict(sorted(provenance["checkpoints"].items()))
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
