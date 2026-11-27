"""Prepare or execute one shard of an invoice processing benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from time import time

from invoice import canonical, clock, elapsed, scheduled


def install_runtime():
    pins = Path("ubuntu-packages.txt").read_text().splitlines()
    rejects = {"tesseract-ocr-ind", "tesseract-ocr-osd"}
    packages = {}
    for name, url, digest in zip(pins[::3], pins[1::3], pins[2::3]):
        if name not in rejects:
            packages.setdefault(name, []).append((url, digest))
    Path("packages").mkdir()

    def fetch(alternatives):
        for url, digest in alternatives:
            output = Path("packages") / Path(url).name
            result = subprocess.run(["curl", "-fsSL", "--retry", "2", "--max-time", "25",
                                     "https://archive.ubuntu.com/ubuntu/" + url, "-o", str(output)])
            if result.returncode == 0 and sha256(output.read_bytes()).hexdigest() == digest:
                return output
        raise RuntimeError(f"No valid Ubuntu package for {alternatives[0][0]}")

    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(fetch, packages.values()))
    for file in files:
        subprocess.run(["dpkg-deb", "-x", str(file), "tess"], check=True)


def shard(root, out, num_nodes, slot):
    raw = (root / "truth.json").read_bytes()
    manifest = json.loads(raw)
    selected = [{k: r[k] for k in ("id", "file", "sha256")} for i, r in enumerate(manifest)
                if i % num_nodes == slot]
    start = clock()
    started_at = time()
    rows = scheduled(root, selected, 2)
    data = {
        "receipt": {"workflow_run": os.environ.get("GITHUB_RUN_ID"), "commit": os.environ.get("GITHUB_SHA"),
                    "nodes": num_nodes, "slot": slot, "cpus_visible": os.cpu_count(), "started_at": started_at,
                    "system": platform.platform(), "ended_at": time(), "ocr_threads": 2,
                    "manifest_sha256": sha256(raw).hexdigest(), **elapsed(start)},
        "rows": rows
    }
    out.write_bytes(canonical(data) + b"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install")
    run = commands.add_parser("shard")
    run.add_argument("input", type=Path)
    run.add_argument("out", type=Path)
    run.add_argument("nodes", type=int)
    run.add_argument("slot", type=int)
    args = parser.parse_args()
    if args.command == "install":
        install_runtime()
    else:
        shard(args.input, args.out, args.nodes, args.slot)

if __name__ == "__main__":
    main()
