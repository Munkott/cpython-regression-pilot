"""Recover the missing suffix of one model-training shard from its durable checkpoint."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import http.client
from multiprocessing import get_context
import os
from pathlib import Path
import signal
import time
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import queue_run as worker
import run
from stream_pilot import REPO, digest, find, label, publish, unpack

PLAN=next(q for q in worker.PLAN["cohorts"] if q["id"]=="fault24")
MODEL=worker.MODEL
STATES=("normal-0","normal-1","fault-first","fault-peer","fault-next")


def assign(role):
    tasks=worker.tasks(PLAN)
    slot=1 if role in ("normal-1","fault-peer") else 0
    part=[task for task in tasks if task["i"]%2==slot]
    if role=="fault-first":
        return part[:PLAN["checkpoint_candidates"]]
    return part


def checkpoint(run_id,commit,release):
    label_name=label(run_id,"checkpoint-fault-first",2,0)
    for tries in range(5):
        try:
            check=find(commit,label_name)
        except (OSError,http.client.HTTPException):
            if tries==4:
                raise
            check=None
        if check:
            break
        if tries==4:
            raise RuntimeError("predecessor checkpoint is unavailable")
        time.sleep(min(8,2**tries))
    blob=check["output"]["summary"]
    assert check["name"]==label_name and check["head_sha"]==commit and check["conclusion"]=="success"
    assert check["app"]["slug"]=="github-actions" and digest(blob)==check["external_id"]
    record=unpack(blob)
    assert record["run"]==run_id and record["commit"]==commit and record["release"]==release
    assert record["plan_sha"]==worker.sweep.PLAN_SHA and record["role"]=="fault-first"
    assert record["executed"]==[task["i"] for task in assign("fault-first")]
    actual={row["i"]:row for row in record["rows"]}
    assert sorted(actual)==record["executed"]
    for task in assign("fault-first"):
        row=actual[task["i"]]
        assert (row["seed"],row["lr"],row["train_seed"])==(task["seed"],task["lr"],task["train_seed"])
    return {"id":check["id"],"digest":check["external_id"],"executed":record["executed"],
            "worker":record["worker"],"selection_samples":record["selection_samples"]}


def choose_missing(role,parent):
    tasks=assign(role)
    if parent is None:
        assert role!="fault-next"
        return tasks
    assert role=="fault-next"
    admitted=parent["executed"]
    planned={row["i"] for row in tasks}
    assert len(admitted)==len(set(admitted)) and set(admitted)<planned
    return [task for task in tasks if task["i"] not in admitted]


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("role",choices=STATES)
    cli.add_argument("release",type=int)
    args=cli.parse_args()
    assert (os.environ["GITHUB_REPOSITORY"],os.environ["GITHUB_EVENT_NAME"])==(REPO,"workflow_dispatch")
    assert os.environ["GITHUB_RUN_ATTEMPT"]=="1"
    run_id,commit=int(os.environ["GITHUB_RUN_ID"]),os.environ["GITHUB_SHA"]
    owner=str(uuid4())
    start=MODEL.clock()
    args.root.mkdir()
    parent=checkpoint(run_id,commit,args.release) if args.role=="fault-next" else None
    members=choose_missing(args.role,parent)
    dataset,wheel=worker.sweep.setup_host(args.root)
    MODEL.load_numpy(wheel,args.root/"site")
    worker.DATASET=dataset
    worker.MODELS=args.root/"models"
    worker.MODELS.mkdir()
    worker.EXCLUDED=worker.excluded(dataset)
    worker.SELECTION=worker.examples(dataset,"selection")
    sample={k:row[2] for k,row in worker.SELECTION.items()}
    if parent:
        assert sample==parent["selection_samples"]
    prepared=run.utc()
    capacity=min(len(members),len(os.sched_getaffinity(0)))
    print(f"FIT_IDS {args.role} "+",".join(str(t["i"]) for t in members),flush=True)
    with ProcessPoolExecutor(max_workers=capacity,mp_context=get_context("fork")) as pool:
        results=list(pool.map(worker.train,members))
    results.sort(key=lambda row:row["i"])
    best=min(results,key=worker.winner_key)
    suffix=f"{args.role}.zip"
    bundle=args.root/suffix
    with ZipFile(bundle,"w",compression=ZIP_DEFLATED,compresslevel=1) as writer:
        writer.write(worker.MODELS/str(best["i"])/worker.sweep.WEIGHT,worker.sweep.WEIGHT)
    assert bundle.stat().st_size<1_500_000
    asset=worker.sweep.full_corpus.upload(args.release,bundle,os.environ["STREAM_CHECK_TOKEN"],commit)
    receipt={"role":args.role,"run":run_id,"commit":commit,"release":args.release,
             "plan_sha":worker.sweep.PLAN_SHA,"worker":owner,"parent":parent,"prepared":prepared,"rows":results,
             "executed":[row["i"] for row in results],"selection_samples":sample,
             "best":best["i"],"asset":asset,"bundle":suffix,"bundle_sha":run.digest(bundle),
             "weights_sha":best["sha"],"bytes":bundle.stat().st_size,"total":MODEL.elapsed(start)}
    role_label="checkpoint-"+args.role
    receipt_id=publish((run_id,commit,role_label,2,0),receipt,title="Model recovery checkpoint")
    if args.role=="fault-first":
        readback=checkpoint(run_id,commit,args.release)
        assert readback["id"]==receipt_id and readback["worker"]==owner
        print(f"CONFIRMED_BEFORE_KILL {receipt_id}",flush=True)
        os.kill(os.getpid(),signal.SIGKILL)


if __name__=="__main__":
    main()
