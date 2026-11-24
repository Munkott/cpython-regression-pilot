"""Compare a full CPython refresh with disjoint hosted work on changed tasks."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from uuid import uuid4

from mixed_pilot import PLAN as MIXED, evaluate, fingerprint
import run
from stream_pilot import REPO, publish

PLAN_PATH = Path(__file__).with_name("refresh_plan.json")
PLAN = json.loads(PLAN_PATH.read_bytes())
SOURCE = PLAN["source"]
RUNTIME = PLAN["runtime"]
TASKS = {task["id"]: task for task in MIXED["tasks"]}
UNITS = sorted({name for task in TASKS.values() for name in task.get("modules", [])} |
               {key for key, task in TASKS.items() if task["kind"] == "inventory"})
TRACK = "refresh"
NODES = 4


def elapsed(start):
    return {"wall_s": time.monotonic() - start[0], "cpu_s": run.cpu() - start[1]}


def prepare(source, out):
    if run.digest(source) != SOURCE["sha256"]:
        raise ValueError("wrong source archive")
    prefix = f"Python-{SOURCE['version']}/Lib/"
    suite = out / "suite"
    suite.mkdir()
    with tarfile.open(source, "r:xz") as archive:
        members = (item.replace(name=item.name.removeprefix(prefix)) for item in archive
                   if item.name.startswith(prefix + "test/"))
        archive.extractall(suite, members=members, filter="data")
    return suite


def unit_result(key, suite, out):
    start = time.monotonic()
    folder = out / key
    if key.startswith("inv-"):
        value = evaluate(TASKS[key], suite, folder, 1)
    else:
        receipt = run.execute(suite, folder, [key], run.PLAN["module_timeout_s"])
        value = {field: receipt[field] for field in ("cases", "exit")}
        value["child_cpu_s"] = receipt.get("child_cpu_s")
    return {"value": value, "wall_s": time.monotonic() - start}


def assemble(results, reuse):
    output = dict(reuse)
    for key, task in TASKS.items():
        if task["kind"] == "inventory":
            if key in results:
                output[key] = results[key]["value"]
        else:
            rows = [results[unit]["value"] for unit in task["modules"]]
            cases = sorted(case for row in rows for case in row["cases"])
            output[key] = {"digest": fingerprint(cases), "outcomes": dict(Counter(case[-1] for case in cases)),
                           "module_exits": [row["exit"] for row in rows]}
    if set(output) != set(TASKS):
        raise ValueError("incomplete assembled output")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--slot", type=int, choices=range(NODES))
    args = parser.parse_args()
    if (args.slot is None) != (args.selection is None):
        parser.error("hosted execution requires a slot and selection")
    if run.digest(run.PLAN_FILE) != PLAN["base_plan_sha256"]:
        raise ValueError("base plan changed")
    if run.digest(Path(__file__).with_name("mixed_plan.json")) != PLAN["mixed_plan_sha256"]:
        raise ValueError("mixed plan changed")
    if ".".join(map(str, sys.version_info[:3])) != SOURCE["version"]:
        raise ValueError("wrong runtime version")
    profile = run.profile()
    if profile["interpreter_sha256"] != RUNTIME["interpreter_sha256"]:
        raise ValueError("wrong interpreter executable")
    start = time.monotonic(), run.cpu()
    out = args.out.resolve()
    out.mkdir()
    suite = prepare(args.source, out)
    setup = elapsed(start)
    if args.slot is None:
        with ThreadPoolExecutor(4) as pool:
            results = dict(zip(UNITS, pool.map(lambda key: unit_result(key, suite, out), UNITS)))
        output = assemble(results, {})
        run.put_json(out / "local.json", {"profile": profile, "setup": setup, "execution": elapsed(start),
                                         "units": results, "tasks": output})
        return
    if (os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_EVENT_NAME")) != (REPO, "workflow_dispatch"):
        raise ValueError("only the public pilot can publish refresh receipts")
    selection = json.loads(args.selection.read_bytes())
    flattened = [key for shard in selection for key in shard]
    if len(selection) != NODES or len(set(flattened)) != len(flattened) or not set(flattened) <= set(UNITS):
        raise ValueError("invalid selection")
    run_id, commit = int(os.environ["GITHUB_RUN_ID"]), os.environ["GITHUB_SHA"]
    common = {"run_id": run_id, "commit": commit, "slot": args.slot, "worker": str(uuid4()),
              "plan_sha256": run.digest(PLAN_PATH), "source_sha256": SOURCE["sha256"],
              "selection_sha256": fingerprint(selection), "profile": profile, "setup": setup,
              "bootstrap": json.loads((args.source.parent / "bootstrap.json").read_bytes())}
    for key in selection[args.slot]:
        index = UNITS.index(key)
        result = unit_result(key, suite, out)
        publish((run_id, commit, TRACK, NODES, index), dict(common, unit=key, result=result))


if __name__ == "__main__":
    main()
