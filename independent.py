"""Execute each assigned CPython module exactly once, with incremental checked receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import run
from stream_pilot import REPO, publish

PLAN_FILE = Path(__file__).with_name("independent_plan.json")


def fingerprint(value):
    payload = json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(",",":")).encode()
    return hashlib.sha256(payload).hexdigest()


def publish_batch(output, identity, packet, rows, local):
    receipt = dict(identity, packet=packet, modules=rows)
    if local:
        run.put_json(output/f"packet-{packet:02d}.json", receipt)
    else:
        tag = f"independent-r{identity['round']}"
        number = 100*identity["slot"] + packet
        publish((identity["run_id"],identity["commit"],tag,identity["nodes"],number),receipt,
                title="Independent CPython regression results")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive",type=Path)
    cli.add_argument("output",type=Path)
    cli.add_argument("--round",type=int,choices=(1,2),default=1)
    cli.add_argument("--nodes",type=int,choices=(1,2,4),default=1)
    cli.add_argument("--slot",type=int,default=0)
    cli.add_argument("--local",action="store_true")
    args = cli.parse_args()
    if not args.local:
        assert os.environ["GITHUB_REPOSITORY"]==REPO and os.environ["GITHUB_EVENT_NAME"]=="workflow_dispatch"
    assert not args.local or (args.nodes,args.slot)==(1,0)
    assert 0 <= args.slot < args.nodes and sys.version_info[:3]==tuple(run.PLAN["interpreter_version"])
    plan = json.loads(PLAN_FILE.read_bytes())
    assert plan["archive_sha256"]==run.PLAN["source_sha256"]
    assigned = plan["allocations"][str(args.nodes)][args.slot]
    capacity = min(len(assigned),len(os.sched_getaffinity(0)))
    output = args.output.resolve()
    output.mkdir()
    started,before = time.monotonic(),run.cpu()
    suite = run.prepare(args.archive.resolve(),output)
    identity = dict(run_id=None if args.local else int(os.environ["GITHUB_RUN_ID"]),
                    commit=None if args.local else os.environ["GITHUB_SHA"],round=args.round,
                    nodes=args.nodes,slot=args.slot,plan_sha256=run.digest(PLAN_FILE),
                    setup_wall_s=time.monotonic()-started,setup_cpu_s=run.cpu()-before,
                    capacity=capacity,prepared_utc=run.utc())
    def execute(name):
        row=run.execute(suite,output/name,[name],run.PLAN["module_timeout_s"])
        actual={"digest":fingerprint(row["cases"]),"cases":len(row["cases"])}
        if not run.succeeded(row) or actual!=plan["expected"][name]:
            raise ValueError(f"{name} disagreed with the frozen reference: exit {row['exit']}, {actual}")
        return dict(actual,name=name,ready=row["finished_utc"],wall_s=row["wall_s"],
                    cpu_s=row["child_cpu_s"])

    with ThreadPoolExecutor(capacity) as pool:
        pool_futures = [pool.submit(execute,name) for name in assigned]
        packet, pending = 0, []
        for future in as_completed(pool_futures):
            pending.append(future.result())
            if len(pending)==plan["packet_size"]:
                publish_batch(output,identity,packet,pending,args.local)
                packet += 1
                pending.clear()
        if pending:
            publish_batch(output,identity,packet,pending,args.local)
    totals = dict(identity, wall_s=time.monotonic()-started, cpu_s=run.cpu()-before,
                  tasks=len(assigned),completed_utc=run.utc())
    run.put_json(output/"total.json",totals)
    print(json.dumps(totals),flush=True)


if __name__=="__main__":
    main()
