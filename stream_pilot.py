"""Run the fixed regression fixture on a bounded stream and publish verified-input receipts."""
import argparse
import base64
import gzip
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import time
from urllib.parse import urlencode
from uuid import uuid4

import run

REPO = "Munkott/cpython-regression-pilot"
PACKETS = 12


def request(method, path, fields=None):
    conn = http.client.HTTPSConnection("api.github.com", timeout=12)
    try:
        conn.request(method, "/repos/" + REPO + path,
                     json.dumps(fields).encode() if fields is not None else None,
                     {"Accept": "application/vnd.github+json", "Content-Type": "application/json",
                      "Authorization": "Bearer " + os.environ["STREAM_CHECK_TOKEN"],
                      "User-Agent": "cpython-stream-pilot"})
        response = conn.getresponse()
        payload = json.loads(response.read(2_000_000))
        if response.status not in (200, 201):
            raise ValueError(f"GitHub Checks HTTP {response.status}: {payload.get('message')}")
        return payload
    finally:
        conn.close()


def label(run_id, track, nodes, packet):
    return f"osi-{run_id}-{track}-n{nodes}-p{packet:02d}"


def find(commit, name):
    query = urlencode({"check_name": name, "filter": "all", "per_page": 5})
    rows = request("GET", f"/commits/{commit}/check-runs?{query}")["check_runs"]
    if len(rows) > 1:
        raise ValueError(f"duplicate check run {name}")
    return rows[0] if rows else None


def pack(receipt):
    blob = gzip.compress(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
    return base64.b64encode(blob).decode("ascii")


def unpack(blob):
    return json.loads(gzip.decompress(base64.b64decode(blob, validate=True)))


def digest(blob):
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def publish(identity, data):
    run_id, commit, track, nodes, packet = identity
    name = label(run_id, track, nodes, packet)
    blob = pack(data)
    if len(blob) > 60_000:
        raise ValueError("receipt exceeds the planned Checks limit")
    body = {"name": name, "head_sha": commit, "status": "completed", "conclusion": "success",
            "completed_at": run.utc(), "external_id": digest(blob),
            "details_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
            "output": {"title": "CPython regression packet receipt", "summary": blob}}
    try:
        result = request("POST", "/check-runs", body)
    except (OSError, http.client.HTTPException):
        # Resolve an ambiguous write through readback before retrying.
        result = None
        for _ in range(2):
            time.sleep(2)
            result = find(commit, name)
            if result:
                break
        if result is None:
            result = request("POST", "/check-runs", body)
    if result["head_sha"] != commit or result["name"] != name or result["external_id"] != body["external_id"]:
        raise ValueError("check write did not preserve the receipt identity")
    print(f"{name} check_id={result['id']}", flush=True)
    return result["id"]


def recovered_seed(run_id, commit, plan_sha):
    name = label(run_id, "recovery", 1, 0)
    checkpoint = find(commit, name)
    if checkpoint is None or checkpoint["conclusion"] != "success" or checkpoint["head_sha"] != commit:
        raise ValueError("successful first checkpoint is unavailable")
    blob = checkpoint["output"]["summary"]
    if checkpoint["external_id"] != digest(blob):
        raise ValueError("checkpoint payload digest mismatch")
    data = unpack(blob)
    rows = data["batch"]["module_receipts"]
    if (data["track"], data["packet"], data["stage"], data["plan_sha256"]) != ("recovery", 0, "seed", plan_sha):
        raise ValueError("checkpoint does not match the recovery task")
    if [row["modules"] for row in rows] != [[module] for module in run.PLAN["modules"]]:
        raise ValueError("checkpoint module inventory differs")
    if not all(map(run.succeeded, rows)):
        raise ValueError("checkpoint contains failing modules")
    return checkpoint["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--nodes", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--stage", choices=["stream", "seed", "resume"], default="stream")
    args = parser.parse_args()
    if os.environ.get("GITHUB_REPOSITORY") != REPO or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise ValueError("the pilot may run only after manual dispatch in its own repository")
    import sys
    if sys.version_info[:3] != tuple(run.PLAN["interpreter_version"]):
        raise ValueError("interpreter version differs from the workload plan")
    if not 0 <= args.slot < args.nodes:
        raise ValueError("invalid slot")
    args.out.mkdir()
    run_id = int(os.environ["GITHUB_RUN_ID"])
    commit = os.environ["GITHUB_SHA"]
    plan_sha = run.digest(run.PLAN_FILE)
    stage = args.stage
    track = "stream" if stage == "stream" else "recovery"
    if stage != "stream" and (args.nodes, args.slot) != (1, 0):
        raise ValueError("recovery uses one runner at a time")
    prior = recovered_seed(run_id, commit, plan_sha) if stage == "resume" else None
    packet_ids = {"stream": range(args.slot, PACKETS, args.nodes), "seed": [0], "resume": [1, 2]}[stage]

    t0, c0 = time.monotonic(), run.cpu()
    suite = run.prepare(args.archive.resolve(), args.out)
    setup = {"wall_s": time.monotonic() - t0, "cpu_s": run.cpu() - c0}
    common = {
        "run_id": run_id, "commit": commit, "track": track, "stage": stage,
        "nodes": args.nodes, "slot": args.slot, "worker": str(uuid4()),
        "plan_sha256": plan_sha, "source_sha256": run.PLAN["source_sha256"],
        "profile": run.profile(), "setup": setup, "prior_checkpoint": prior,
    }
    for packet in packet_ids:
        start = run.utc()
        batch = run.run_batch(suite, args.out / f"packet-{packet:02d}",
                              run.PLAN["modules"], 1, False)
        if not all(map(run.succeeded, batch["module_receipts"])):
            raise ValueError(f"module failed in packet {packet}")
        receipt = dict(common, packet=packet, started_utc=start, finished_utc=run.utc(), batch=batch)
        accepted_id = publish((run_id, commit, track, args.nodes, packet), receipt)
        if stage == "seed":
            run.put_json(args.out / "published.json", {"check_id": accepted_id, "packet": packet})
            signal.pause()
            raise RuntimeError("seed was expected to be terminated")


if __name__ == "__main__":
    main()
