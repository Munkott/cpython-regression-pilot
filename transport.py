"""Run one shard of q144 and return one artifact with its scores and candidate weights."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil

import queue_run as queue
import run
import sweep

MODEL = queue.MODEL
EXPECTED_REPOSITORY = "Munkott/cpython-regression-pilot"


def prepare(root, stage):
    start = MODEL.clock()
    root.mkdir()
    dataset, wheel = sweep.setup_host(root)
    # Keep the validated corpus and wheel together for every runner.
    stage.mkdir()
    shutil.copytree(dataset, stage / "dataset")
    shutil.copyfile(wheel, stage / "numpy.whl")
    assert run.digest(stage / "dataset" / "manifest.json") == MODEL.PLAN["corpus"]["manifest_sha256"]
    run.put_json(stage / "prepare.json", {
        "time": MODEL.elapsed(start), "dataset": run.digest(dataset / "manifest.json"),
        "wheel": run.digest(wheel), "finished": run.utc(),
    })


def worker(path, prepared, output, count, slot):
    if (os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_EVENT_NAME")) != (EXPECTED_REPOSITORY, "push"):
        raise ValueError("Unexpected hosted workflow")
    if not 0 <= slot < count:
        raise ValueError("Slot out of range")
    path.mkdir()
    output.mkdir()
    start = MODEL.clock()
    dataset = prepared / "dataset"
    MODEL.load_numpy(prepared / "numpy.whl", path / "site")
    queue.DATASET = dataset
    queue.EXCLUDED = queue.excluded(dataset)
    queue.SELECTION = queue.examples(dataset, "selection")
    queue.MODELS = path / "models"
    queue.MODELS.mkdir()
    cohort, = [row for row in queue.PLAN["cohorts"] if row["id"] == "q144"]
    assigned = [task for task in queue.tasks(cohort) if task["i"] % count == slot]
    capacity = min(len(assigned), len(os.sched_getaffinity(0)))
    setup = MODEL.elapsed(start)
    begun = run.utc()
    rows = []
    with ProcessPoolExecutor(max_workers=capacity, mp_context=get_context("fork")) as pool:
        futures = [pool.submit(queue.train, job) for job in assigned]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"completed candidate {row['i']} of this queue ({len(rows)}/{len(assigned)})", flush=True)
    best = min(rows, key=queue.winner_key)
    checkpoint = queue.MODELS / str(best["i"]) / sweep.WEIGHT
    shutil.copyfile(checkpoint, output / sweep.WEIGHT)
    assert run.digest(output / sweep.WEIGHT) == best["sha"]
    run.put_json(output / "receipt.json", {
        "run": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
        "nodes": count, "slot": slot, "queue": cohort["id"], "plan_sha": sweep.PLAN_SHA,
        "capacity": capacity, "setup": setup, "started": begun, "finished": run.utc(),
        "rows": sorted(rows, key=lambda row: row["i"]), "best": best["i"], "total": MODEL.elapsed(start),
        "selection_samples": {key: value[2] for key,value in queue.SELECTION.items()},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="mode", required=True)
    prep = actions.add_parser("prepare")
    prep.add_argument("root", type=Path)
    prep.add_argument("out", type=Path)
    job = actions.add_parser("worker")
    job.add_argument("root", type=Path)
    job.add_argument("prepared", type=Path)
    job.add_argument("out", type=Path)
    job.add_argument("count", type=int, choices=(1,2,4,8))
    job.add_argument("slot", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root, args.out)
    else:
        worker(args.root, args.prepared, args.out, args.count, args.slot)
