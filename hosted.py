"""Prepare independent shards and validate their receipt for the index benchmark."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import platform
import subprocess
import time

from index import (PAIRS, collect, compute_chunk, digest, elapsed, manifest, merge_results,
                   now, prior_cache, read_blobs, weighted_bins, write)


PHASES = {
    "small1": ("small", 1), "large1": ("large", 1),
    "large2": ("large", 2), "large4": ("large", 4), "large8": ("large", 8),
    "large1b": ("large", 1),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_bytes())


def make_plan(repo):
    plan = {}
    for phase, (load_name, workers) in PHASES.items():
        base, head = PAIRS[load_name]
        old, _ = manifest(repo, base)
        paths, sizes = manifest(repo, head)
        needed = set(paths.values()) - set(old.values())
        plan[phase] = {"base": base, "head": head, "bins": weighted_bins(needed, sizes, workers)}
    return plan


def prepare(repo, out, plan):
    start = now()
    out.mkdir()
    for phase, config in plan.items():
        for slot, oids in enumerate(config["bins"]):
            rows = read_blobs(repo, oids)
            destination = out / f"{phase}-{slot}.input.gz"
            with gzip.open(destination, "wt", compresslevel=4) as stream:
                json.dump([(oid, body.hex()) for oid, body in rows], stream, separators=(",", ":"))
    write(out / "setup.json", {"elapsed": elapsed(start), "timestamp": utc()})


def worker(input_dir, result_dir, phase, slot, plan):
    start = now()
    began = utc()
    config = plan[phase]
    with gzip.open(input_dir / f"{phase}-{slot}.input.gz", "rt") as stream:
        rows = [(oid, bytes.fromhex(body)) for oid, body in json.load(stream)]
    if [oid for oid, _ in rows] != config["bins"][slot]:
        raise ValueError("Unexpected shard content")
    records = compute_chunk(rows)
    result_dir.mkdir(exist_ok=True)
    write(result_dir / "receipt.json", {
        "phase": phase, "slot": slot, "head": config["head"],
        "records": records, "digest": digest(records),
        "started": began, "finished": utc(), "elapsed": elapsed(start),
        "cpu": platform.processor(), "host": platform.node(),
        "raw_bytes": sum(len(body) for _, body in rows),
    })


def verify(repo, phase, plan, receipts, expected):
    start = now()
    config = plan[phase]
    paths, _ = manifest(repo, config["head"])
    _, cached, _, setup = prior_cache(repo, config["base"])
    # Charge only merge/serialization to the warm accept path.
    start = now()
    deliveries = []
    metrics = []
    for file in sorted(receipts.rglob("receipt.json")):
        row = load(file)
        if row["phase"] != phase:
            continue
        terms = row.pop("records")
        if row["head"] != config["head"] or digest(terms) != row["digest"]:
            raise ValueError(f"Bad receipt in {file}")
        deliveries.append((row["slot"], terms))
        metrics.append(row)
    records, missing, duplicates = merge_results(config["bins"], deliveries, cached)
    if missing or duplicates:
        raise ValueError(f"Unexpected receipt set. Missing {missing}, repeated {duplicates}")
    product = collect(paths, records, config["head"])
    fingerprint = digest(product)
    assert fingerprint == expected, phase
    return {"phase": phase, "sha256": fingerprint, "workers": metrics,
            "pc_accept": elapsed(start), "parent_setup": setup}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plan", "prepare", "worker", "verify"])
    parser.add_argument("--repo", type=Path, default=Path("repository.git"))
    parser.add_argument("--in", dest="inputs", type=Path, default=Path("inputs"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--phase")
    parser.add_argument("--slot", type=int)
    parser.add_argument("--expected")
    args = parser.parse_args()
    path = Path(__file__).with_name("hosted-plan.json")
    if args.mode == "plan":
        if path.exists():
            raise FileExistsError(path)
        write(path, make_plan(args.repo))
    else:
        plan = load(path)
        if args.mode == "prepare":
            prepare(args.repo, args.inputs, plan)
        elif args.mode == "worker":
            worker(args.inputs, args.out, args.phase, args.slot, plan)
        elif args.mode == "verify":
            print(json.dumps(verify(args.repo, args.phase, plan, args.inputs, args.expected)))
