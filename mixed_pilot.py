"""Execute the frozen regression and archive-inventory workload."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import run
from stream_pilot import REPO, publish

PLAN_FILE = Path(__file__).with_name("mixed_plan.json")
PLAN = json.loads(PLAN_FILE.read_bytes())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def evaluate(task, suite, folder, workers):
    if task["kind"] == "regression":
        batch = run.run_batch(suite, folder, task["modules"], workers, False)
        if not all(map(run.succeeded,batch["module_receipts"])):
            raise ValueError(f"regression failed in {task['id']}")
        cases = sorted(case for item in batch["module_receipts"] for case in item["cases"])
        return {"digest":fingerprint(cases),"outcomes":dict(Counter(case[-1] for case in cases))}
    if task["kind"] != "inventory":
        raise ValueError("unknown task")
    root = suite / "test" / task["folder"]
    files = [[path.relative_to(root).as_posix(),path.stat().st_size,run.digest(path)]
             for path in root.rglob("*") if not path.is_symlink() and path.is_file()]
    if not files:
        raise ValueError("empty inventory")
    return {"digest":fingerprint(sorted(files)),"files":len(files),"body_bytes":sum(row[1] for row in files)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--round",type=int,choices=(1,2),default=1)
    p.add_argument("--nodes",type=int,choices=(1,2,4),default=1)
    p.add_argument("--slot",type=int,default=0)
    p.add_argument("--local",action="store_true")
    args=p.parse_args()
    if run.digest(run.PLAN_FILE)!=PLAN["base_plan_sha256"]:
        raise ValueError("the reference workload changed")
    if sys.version_info[:3]!=tuple(run.PLAN["interpreter_version"]):
        raise ValueError("wrong CPython runtime")
    if not 0<=args.slot<args.nodes:
        raise ValueError("invalid worker slot")
    if args.local and (args.nodes,args.slot)!=(1,0):
        raise ValueError("local control executes the whole workload")
    if not args.local and (os.environ.get("GITHUB_REPOSITORY")!=REPO or
                           os.environ.get("GITHUB_EVENT_NAME")!="workflow_dispatch"):
        raise ValueError("only the pilot workflow may publish results")

    args.out.mkdir()
    t,c=time.monotonic(),run.cpu()
    suite=run.prepare(args.archive.resolve(),args.out)
    setup={"wall_s":time.monotonic()-t,"cpu_s":run.cpu()-c}
    workflow_id = int(os.environ["GITHUB_RUN_ID"]) if not args.local else None
    commit = os.environ["GITHUB_SHA"] if not args.local else None
    track=f"mixed-r{args.round}"
    identity={"setup":setup,"plan_sha256":run.digest(PLAN_FILE),"source_sha256":run.PLAN["source_sha256"],
              "round":args.round,"nodes":args.nodes,"slot":args.slot,"worker":str(uuid4()),
              "profile":run.profile(),"run_id":workflow_id,"commit":commit}
    definitions={task["id"]:(index,task) for index,task in enumerate(PLAN["tasks"])}
    reports=[]
    t0,c0=time.monotonic(),run.cpu()
    for key in PLAN["allocations"][str(args.nodes)][args.slot]:
        index,task=definitions[key]
        t,c=time.monotonic(),run.cpu()
        result=evaluate(task,suite,args.out/key,4 if args.local else 1)
        receipt=dict(identity,task=key,kind=task["kind"],index=index,result=result,
                     timing={"wall_s":time.monotonic()-t,"cpu_s":run.cpu()-c},
                     finished_utc=run.utc())
        if args.local:
            reports.append(receipt)
            print(f"Local {len(reports)}/32 {key}",flush=True)
        else:
            publish((workflow_id,commit,track,args.nodes,index),receipt)
    if args.local:
        run.put_json(args.out/"local.json",{"worker_processes":4,"setup":setup,
                     "work":{"wall_s":time.monotonic()-t0,"cpu_s":run.cpu()-c0},"receipts":reports})


if __name__=="__main__":
    main()
