"""Fit fixed independent queues on one runner with a reusable process pool."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import time

import queue_run as queue
import run
import sweep
from transport import EXPECTED_REPOSITORY

COHORTS = ("q144", "repeat144", "q72")
MODEL = queue.MODEL


def train(catalog, task):
    queue.MODELS = catalog
    return queue.train(task)


def train_queues(prepared, out, slot, schedule, run_hosted=True):
    if not 0 <= slot < 8:
        raise ValueError("Invalid fixed slot")
    identity = (os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_EVENT_NAME"))
    if run_hosted and identity != (EXPECTED_REPOSITORY, "push"):
        raise ValueError("Unexpected workflow identity")
    begin = MODEL.clock()
    out.mkdir(exist_ok=True)
    queue.DATASET = prepared / "dataset"
    MODEL.load_numpy(prepared / "numpy.whl", out / "site")
    queue.EXCLUDED = queue.excluded(queue.DATASET)
    queue.SELECTION = queue.examples(queue.DATASET, "selection")
    capacity = min(4, len(os.sched_getaffinity(0)))
    setup = MODEL.elapsed(begin)
    worker_started = run.utc()
    with ProcessPoolExecutor(max_workers=capacity, mp_context=get_context("fork")) as pool:
        for qid in schedule:
            row_start = time.monotonic()
            started = run.utc()
            config, = (row for row in queue.PLAN["cohorts"] if row["id"] == qid)
            assigned = [task for task in queue.tasks(config) if task["i"] % 8 == slot]
            unit = out / qid
            catalog = unit / "models"
            catalog.mkdir(parents=True)
            scores = unit / "scores"
            weights = unit / "weights"
            scores.mkdir()
            weights.mkdir()
            futures = [pool.submit(train, catalog, task) for task in assigned]
            rows = [job.result() for job in as_completed(futures)]
            best = min(rows, key=queue.winner_key)
            target = weights / sweep.WEIGHT
            shutil.copyfile(catalog / str(best["i"]) / sweep.WEIGHT, target)
            if run.digest(target) != best["sha"]:
                raise ValueError("Checkpoint copy digest differed")
            run.put_json(scores / "receipt.json", {
                "run": int(os.environ.get("GITHUB_RUN_ID", 0)), "commit": os.environ.get("GITHUB_SHA"),
                "queue": qid, "slot": slot, "capacity": capacity, "plan_sha": sweep.PLAN_SHA,
                "worker_started": worker_started, "worker_setup": setup, "worker_pid": os.getpid(),
                "started": started, "finished": run.utc(), "queue_wall_s": time.monotonic()-row_start,
                "rows": sorted(rows, key=lambda row: row["i"]), "best": best["i"],
                "selection_samples": {k: item[2] for k,item in queue.SELECTION.items()},
            })
            print(qid, slot, len(rows), "ready", flush=True)


def wait_for(out, qid):
    deadline = time.monotonic() + 660
    receipt = out / qid / "scores" / "receipt.json"
    pid = int((out / "worker.pid").read_text())
    while not receipt.is_file():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise RuntimeError("The persistent worker exited before completing the requested queue")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Worker did not produce {qid} within the job budget")
        time.sleep(.5)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    work = sub.add_parser("work")
    work.add_argument("prepared", type=Path)
    work.add_argument("output", type=Path)
    work.add_argument("slot", type=int)
    work.add_argument("schedule", choices=COHORTS, nargs="+")
    wait = sub.add_parser("wait")
    wait.add_argument("output", type=Path)
    wait.add_argument("qid", choices=COHORTS)
    args = cli.parse_args()
    if args.command == "work":
        train_queues(args.prepared, args.output.resolve(), args.slot, args.schedule)
    else:
        wait_for(args.output, args.qid)
