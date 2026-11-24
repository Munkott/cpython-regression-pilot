"""Prepare a reproducible token-window dataset from the public NLTK Gutenberg corpus."""
import argparse
from array import array
from base64 import b64decode, b64encode
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unicodedata
from uuid import uuid4
from zipfile import ZipFile

import run
from stream_pilot import REPO, publish

PLAN_PATH = Path(__file__).with_name("corpus_plan.json")
PLAN = json.loads(PLAN_PATH.read_bytes())
DOCS = PLAN["documents"]
WIDTH = PLAN["window_tokens"]
TRACK = "corpus"
NODES = 4


def timing(start):
    return {"wall_s": time.monotonic() - start[0], "cpu_s": run.cpu() - start[1]}


def start():
    return time.monotonic(), run.cpu()


def fetch_inputs(cache):
    for name, spec in PLAN["inputs"].items():
        path = cache / name
        for url in spec["urls"]:
            headers = (["-H", "Accept: application/vnd.github.raw+json"]
                       if url.startswith("https://api.github.com/") else [])
            command = ["curl", "-4", "-L", "--fail", "--silent", "--show-error", "--proto", "=https",
                       "--connect-timeout", "10", "--max-time", "35", "--retry", "1",
                       "--retry-max-time", "44", "--max-filesize", str(spec["size"]), *headers,
                       "-o", str(path), url]
            if subprocess.run(command, check=False).returncode == 0 and run.digest(path) == spec["sha256"]:
                break
        else:
            raise ValueError(f"could not acquire verified input {name}")


def prepare(cache, out):
    for name, spec in PLAN["inputs"].items():
        path = cache / name
        if path.stat().st_size != spec["size"] or run.digest(path) != spec["sha256"]:
            raise ValueError(f"invalid pinned input {name}")
    site = out / "site"
    site.mkdir()
    for name in ("tiktoken.whl", "regex.whl"):
        with ZipFile(cache / name) as wheel:
            wheel.extractall(site)
    sys.path.insert(0, str(site))
    import tiktoken

    encoding_cache = out / "bpe"
    encoding_cache.mkdir()
    url = PLAN["inputs"]["r50k_base.tiktoken"]["urls"][0]
    entry = hashlib.sha1(url.encode()).hexdigest()
    (encoding_cache / entry).write_bytes((cache / "r50k_base.tiktoken").read_bytes())
    os.environ["TIKTOKEN_CACHE_DIR"] = str(encoding_cache)
    encoding = tiktoken.get_encoding("r50k_base")
    if tiktoken.__version__ != "0.9.0" or encoding.n_vocab != 50_257 or array("H").itemsize != 2:
        raise ValueError("unexpected tokeniser or token format")
    return encoding


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def process(cache, encoder, doc):
    started = time.monotonic()
    cpu_before = time.thread_time()
    with ZipFile(cache / "gutenberg-api.zip") as corpus:
        raw = corpus.read(doc["entry"])
    if len(raw) != doc["size"] or checksum(raw) != doc["sha256"]:
        raise ValueError("wrong document bytes")
    text = raw.decode(doc["codec"]).replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text).strip()
    ids = array("H", encoder.encode_ordinary(text))
    if sys.byteorder != "little":
        ids.byteswap()
    blocks = len(ids) // WIDTH
    if blocks < PLAN["samples_per_document"]:
        raise ValueError("document too short for the fixed sampling plan")
    windows = []
    for n in range(PLAN["samples_per_document"]):
        offset = (n * (blocks - 1) // (PLAN["samples_per_document"] - 1)) * WIDTH
        payload = ids[offset:offset + WIDTH].tobytes()
        windows.append({"offset_tokens": offset, "content_u16le": b64encode(payload).decode("ascii")})
    result = {"document": doc["entry"], "input_sha256": doc["sha256"],
              "unicode_sha256": checksum(text.encode("utf-8")), "token_sha256": checksum(ids.tobytes()),
              "input_bytes": len(raw), "token_count": len(ids), "split": doc["split"], "windows": windows}
    return {"result": result, "time": {"wall_s": time.monotonic() - started,
                                       "thread_cpu_s": time.thread_time() - cpu_before}}


def assemble(rows, out):
    if set(rows) != {doc["entry"] for doc in DOCS}:
        raise ValueError("incomplete document coverage")
    split_bytes = {"train": bytearray(), "validation": bytearray()}
    seen = set()
    outputs = []
    documents = []
    # Keep validation samples first if content repeats across splits.
    for doc in sorted(DOCS, key=lambda row: (row["split"] == "train", row["entry"])):
        key = doc["entry"]
        row = rows[key]
        if row["document"] != key or row["split"] != doc["split"]:
            raise ValueError("document or split differs from plan")
        documents.append({name: val for name, val in row.items() if name != "windows"})
        for window in row["windows"]:
            data = b64decode(window["content_u16le"], validate=True)
            if len(data) != WIDTH * 2:
                raise ValueError("invalid window length")
            digest = checksum(data)
            if digest in seen:
                continue
            seen.add(digest)
            split = row["split"]
            target = split_bytes[split]
            outputs.append({"document": key, "batch": doc["batch"], "split": split,
                            "source_offset_tokens": window["offset_tokens"],
                            "byte_offset": len(target), "sha256": digest})
            target.extend(data)
    out.mkdir()
    splits = {}
    for split, data in split_bytes.items():
        path = out / f"{split}.u16"
        with path.open("wb") as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
        splits[split] = {"path": path.name, "tokens": len(data) // 2, "sha256": checksum(data)}
    manifest = {"encoding": "r50k_base", "dtype": "uint16-le", "context_tokens": WIDTH,
                "documents": documents, "windows": outputs, "splits": splits,
                "exact_duplicate_windows_removed": len(DOCS) * PLAN["samples_per_document"] - len(outputs)}
    run.put_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--slot", type=int, choices=range(NODES))
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.slot is not None and (os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_EVENT_NAME")) != (
            REPO, "workflow_dispatch"):
        raise ValueError("hosted mode belongs to the dedicated public pilot")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("the pinned wheels require CPython 3.12")
    tick = start()
    args.out.mkdir()
    if args.fetch:
        args.cache.mkdir()
        fetch_inputs(args.cache)
    encoder = prepare(args.cache, args.out)
    setup = timing(tick)
    print(f"Prepared input in {setup['wall_s']:.3f} s", flush=True)
    if args.slot is None:
        keys = sorted(DOCS, key=lambda row: -row["size"])
        with ThreadPoolExecutor(4) as pool:
            computed = list(pool.map(lambda doc: process(args.cache, encoder, doc), keys))
        results = {item["result"]["document"]: item for item in computed}
        assemble({key: value["result"] for key, value in results.items()}, args.out / "dataset")
        run.put_json(args.out / "local.json", {"setup": setup, "total": timing(tick),
                                             "documents": results, "platform": run.profile()})
        return
    run_id, commit = int(os.environ["GITHUB_RUN_ID"]), os.environ["GITHUB_SHA"]
    identity = {"run_id": run_id, "commit": commit, "plan_sha256": run.digest(PLAN_PATH),
                "slot": args.slot, "worker": str(uuid4()), "setup": setup}
    for index, task in enumerate(DOCS):
        if task["slot"] != args.slot:
            continue
        result = process(args.cache, encoder, task)
        publish((run_id, commit, TRACK, NODES, index), dict(identity, index=index, data=result),
                title="Gutenberg token-window receipt")


if __name__ == "__main__":
    main()
