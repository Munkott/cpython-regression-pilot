"""Execute a fixed CPython regression workload on one Linux host."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET

PLAN_FILE = Path(__file__).with_name("plan.json")
PLAN = json.loads(PLAN_FILE.read_bytes())
LAUNCHER = "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module('test', run_name='__main__')"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu():
    usage = [resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(item.ru_utime + item.ru_stime for item in usage)


def put_json(path, value):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def exclusive(out):
    with (out / ".lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepare(archive, out):
    if digest(archive) != PLAN["source_sha256"]:
        raise ValueError("wrong source archive")
    suite = out / "suite"
    suite.mkdir(exist_ok=True)
    prefix = "Python-3.12.3/Lib/"
    with tarfile.open(archive, mode="r:xz") as source:
        members = [member.replace(name=member.name.removeprefix(prefix))
                   for member in source if member.name.startswith(prefix + "test/")]
        source.extractall(path=suite, members=members, filter="data")
    return suite


def profile():
    return {
        "interpreter": sys.version,
        "interpreter_sha256": digest(Path(sys.executable).resolve()),
        "libc": list(platform.libc_ver()),
        "machine": platform.machine(),
        "available_logical_cpus": len(os.sched_getaffinity(0)),
        "platform": platform.platform(),
        "os": platform.freedesktop_os_release().get("PRETTY_NAME"),
    }


def read_cases(xml):
    root = ET.parse(xml).getroot()
    outcomes = []
    for case in root.iter("testcase"):
        status = next((label for label in ("error", "failure", "skipped")
                       if case.find(label) is not None), "pass")
        outcomes.append([case.get("classname", ""), case.attrib["name"], status])
    return sorted(outcomes)


def execute(suite, folder, modules, timeout):
    folder.mkdir(exist_ok=True)
    xml = folder / "junit.xml"
    xml.unlink(missing_ok=True)
    counter = folder / "cpu.txt"
    counter.unlink(missing_ok=True)
    cmd = ["/usr/bin/time", "-f", "%U %S", "-o", str(counter), sys.executable,
           "-I", "-B", "-c", LAUNCHER, str(suite),
           "--junit-xml", str(xml), "--randseed", str(PLAN["random_seed"]), *modules]
    env = {"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC",
           "HOME": str(folder), "TMPDIR": str(folder)}
    started = time.monotonic()
    with (folder / "output.log").open("wb") as log:
        with subprocess.Popen(cmd, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    row = {
        "modules": modules, "exit": process.returncode,
        "wall_s": time.monotonic() - started, "finished_utc": utc(),
        "cases": read_cases(xml) if xml.is_file() else [],
    }
    if counter.is_file() and process.returncode == 0:
        row["child_cpu_s"] = sum(map(float, counter.read_text().split()))
    put_json(folder / "receipt.json", row)
    return row


def succeeded(row):
    statuses = {case[-1] for case in row["cases"]}
    return row["exit"] == 0 and "pass" in statuses and statuses <= {"pass", "skipped"}


def run_batch(suite, output, modules, workers, resume):
    output.mkdir(exist_ok=True)

    def run_one(module):
        folder = output / module
        receipt = folder / "receipt.json"
        if resume and receipt.is_file():
            previous = json.loads(receipt.read_text())
            if previous["modules"] == [module] and succeeded(previous):
                return previous, True
        return execute(suite, folder, [module], PLAN["module_timeout_s"]), False

    start, before = time.monotonic(), cpu()
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(run_one, modules))
    return {
        "workers": workers, "wall_s": time.monotonic() - start, "cpu_s": cpu() - before,
        "reused_receipts": sum(reused for _, reused in results),
        "module_receipts": [row for row, _ in results],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--campaign", action="store_true", help="run all planned local comparisons")
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2, 4))
    parser.add_argument("--shards", type=int, default=1, choices=(1, 2, 4))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue the same single batch")
    args = parser.parse_args()
    if sys.version_info[:3] != tuple(PLAN["interpreter_version"]):
        parser.error("interpreter version differs from the plan")
    if not 0 <= args.shard_index < args.shards:
        parser.error("shard index out of range")
    if args.campaign and (args.shards != 1 or args.workers != 1 or args.resume):
        parser.error("campaign follows the worker order in the plan")
    out = args.out.resolve()
    out.mkdir(exist_ok=args.resume)
    modules = PLAN["modules"][args.shard_index::args.shards]
    with exclusive(out):
        identity = {"plan_sha256": digest(PLAN_FILE), "source_sha256": PLAN["source_sha256"],
                    "profile": profile(), "shards": args.shards, "shard_index": args.shard_index,
                    "workers": args.workers, "modules": modules}
        binding = out / "identity.json"
        if binding.is_file() and json.loads(binding.read_text()) != identity:
            raise ValueError("resume identity differs from the initial run")
        put_json(binding, identity)
        start, before = time.monotonic(), cpu()
        suite = prepare(args.archive.resolve(), out)
        setup = {"wall_s": time.monotonic() - start, "cpu_s": cpu() - before}
        result = {"identity": identity, "setup": setup, "runs": []}
        if args.campaign:
            reference = execute(suite, out / "reference", modules, 90)
            result["reference"] = reference
            print(f"stock reference: exit {reference['exit']}, {len(reference['cases'])} cases", flush=True)
            if not succeeded(reference):
                put_json(out / "result.json", result)
                return 1
            configurations = [workers for order in PLAN["local_worker_orders"] for workers in order]
        else:
            configurations = [args.workers]
        for index, workers in enumerate(configurations):
            if time.monotonic() - start + PLAN["module_timeout_s"] > PLAN["campaign_budget_s"]:
                break
            batch = run_batch(suite, out / f"batch-{index}", modules, workers, args.resume)
            result["runs"].append(batch)
            put_json(out / "result.json", result)
            print(f"{workers} workers: {batch['wall_s']:.3f}s wall, {batch['cpu_s']:.3f}s CPU", flush=True)
            if not all(map(succeeded, batch["module_receipts"])):
                return 1
        return int(len(result["runs"]) != len(configurations))


if __name__ == "__main__":
    sys.exit(main())
