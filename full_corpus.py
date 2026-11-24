"""Materialize all token sequences for the pinned Gutenberg and Europarl documents."""
import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
import time
import unicodedata
from urllib.parse import urlencode
from uuid import uuid4
from zipfile import ZipFile

import corpus_pilot
import run
from stream_pilot import REPO, publish

PLAN_PATH = Path(__file__).with_name("full_corpus_plan.json")
PLAN = json.loads(PLAN_PATH.read_bytes())
DOCS = PLAN["documents"]
TRACK = "corpus-full"
FILES = {"train": "train.u16", "validation": "validation.u16"}


def doc_ids(nodes, slot):
    return [row["index"] for row in DOCS if nodes == 1 or row["slot"] == slot]


def process(doc, cache, encoder, out):
    with ZipFile(cache / doc["source"]) as source:
        raw = source.read(doc["entry"])
    if len(raw) != doc["size"] or hashlib.sha256(raw).hexdigest() != doc["sha256"]:
        raise ValueError("document differs from pinned source")
    text = raw.decode(doc["codec"]).replace("\r\n", "\n").replace("\r", "\n")
    if doc["source"] == "europarl-raw.zip":
        text = "\n".join(line for line in text.split("\n") if not line.lstrip().startswith("<"))
    text = unicodedata.normalize("NFC", text).strip()
    if not text:
        raise ValueError("document normalized to empty text")
    tokens = array("H", encoder.encode_ordinary(text))
    tokens.append(PLAN["eot"])
    if sys.byteorder != "little":
        tokens.byteswap()
    body = tokens.tobytes()
    path = out / file_name(doc["index"])
    path.write_bytes(body)
    return {"index": doc["index"], "tokens": len(tokens),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "tokens_sha256": hashlib.sha256(body).hexdigest()}


def file_name(index):
    return f"doc-{index:03d}.u16"


def create_dataset(rows, basedir, out):
    if set(rows) != set(range(len(DOCS))):
        raise ValueError("document roster is incomplete")
    out.mkdir()
    outputs = {}
    details = []
    for split, name in FILES.items():
        count = 0
        target = out / name
        with target.open("wb") as writer:
            for doc in DOCS:
                if doc["split"] != split:
                    continue
                index = doc["index"]
                result = rows[index]
                path = basedir / file_name(index)
                body = path.read_bytes()
                if (result["index"] != index or
                        len(body) != 2 * result["tokens"] or
                        hashlib.sha256(body).hexdigest() != result["tokens_sha256"]):
                    raise ValueError("token result is inconsistent")
                if body[-2:] != PLAN["eot"].to_bytes(2, "little"):
                    raise ValueError("missing document boundary token")
                writer.write(body)
                details.append({"document":doc["entry"], "source":doc["source"], "batch":doc["batch"],
                                "input_sha256":doc["sha256"], "input_bytes":doc["size"],
                                "split":split, "offset_tokens":count, **result})
                count += result["tokens"]
            writer.flush()
            os.fsync(writer.fileno())
        outputs[split] = {"tokens":count, "bytes":target.stat().st_size, "file":name, "sha256":run.digest(target)}
    manifest = {"dtype":PLAN["dtype"], "encoding":PLAN["encoding"], "eot":PLAN["eot"],
                "plan_sha256":run.digest(PLAN_PATH), "splits":outputs, "documents":details}
    run.put_json(out / "manifest.json", manifest)
    return manifest


def pack(folder, rows, out):
    with ZipFile(out, "w", compression=8, compresslevel=1) as archive:
        archive.writestr("documents.json", json.dumps(rows, sort_keys=True))
        for index in sorted(rows):
            name = file_name(index)
            archive.write(folder / name, name)
    if out.stat().st_size > PLAN["max_bundle_bytes"]:
        raise ValueError("bundle exceeds the experimental bound")


def release_data(release_id, token):
    conn = http.client.HTTPSConnection("api.github.com", timeout=12)
    try:
        conn.request("GET", f"/repos/{REPO}/releases/{release_id}", headers={
            "Authorization":"Bearer " + token,"Accept":"application/vnd.github+json","User-Agent":"public-corpus-pilot"})
        response = conn.getresponse()
        if response.status != 200:
            raise ValueError(f"release lookup HTTP {response.status}")
        return json.loads(response.read(400_001))
    finally:
        conn.close()


def upload(release_id, path, token, expected_commit):
    release = release_data(release_id, token)
    if (not release["draft"] or release["id"] != release_id or release["target_commitish"] != expected_commit
            or release["tag_name"] != f"full-corpus-{expected_commit[:12]}"):
        raise ValueError("unexpected draft release")
    if any(asset["name"] == path.name for asset in release["assets"]):
        raise ValueError("bundle already present in this draft")
    conn = http.client.HTTPSConnection("uploads.github.com", timeout=40)
    try:
        resource = f"/repos/{REPO}/releases/{release_id}/assets?" + urlencode({"name":path.name})
        with path.open("rb") as reader:
            conn.request("POST", resource, body=reader, headers={
                "Authorization": "Bearer " + token, "User-Agent": "public-corpus-pilot",
                "Content-Type": "application/zip", "Content-Length": str(path.stat().st_size)})
        response = conn.getresponse()
        data = json.loads(response.read(100_001))
        if response.status != 201:
            raise ValueError(f"bundle upload HTTP {response.status}")
        if data["name"] != path.name or data["size"] != path.stat().st_size:
            raise ValueError("uploaded bundle has wrong identity")
        return data["id"]
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--nodes", type=int, choices=(1,4))
    parser.add_argument("--slot", type=int, choices=range(4))
    parser.add_argument("--release", type=int)
    args = parser.parse_args()
    hosted = args.nodes is not None
    if hosted and (args.slot is None or not 0 <= args.slot < args.nodes or args.release is None):
        parser.error("hosted mode requires a release and valid slot")
    if hosted and (os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_EVENT_NAME")) != (REPO, "workflow_dispatch"):
        raise ValueError("only the designated public workflow may upload")
    t0 = corpus_pilot.start()
    args.out.mkdir()
    if hosted:
        args.cache.mkdir()
        corpus_pilot.fetch_inputs(args.cache, PLAN["inputs"])
    encoder = corpus_pilot.prepare(args.cache, args.out, PLAN["inputs"])
    setup = corpus_pilot.timing(t0)
    indices = doc_ids(args.nodes or 1, args.slot or 0)
    ordered = sorted(indices, key=lambda index: -DOCS[index]["size"])
    interim = args.out / "parts"
    interim.mkdir()
    tick = corpus_pilot.start()
    with ThreadPoolExecutor(4) as pool:
        pieces = pool.map(lambda index: process(DOCS[index], args.cache, encoder, interim), ordered)
        rows = {item["index"]:item for item in pieces}
    compute = corpus_pilot.timing(tick)
    if not hosted:
        create_dataset(rows, interim, args.out / "dataset")
        run.put_json(args.out / "local.json", {"total":corpus_pilot.timing(t0),
                     "setup":setup, "compute":compute, "profile":run.profile()})
        return
    tick = corpus_pilot.start()
    name = f"n{args.nodes}-slot{args.slot}.zip"
    bundle = args.out / name
    pack(interim, rows, bundle)
    bundle_sha = run.digest(bundle)
    packing = corpus_pilot.timing(tick)
    run_id, commit = int(os.environ["GITHUB_RUN_ID"]), os.environ["GITHUB_SHA"]
    tick = corpus_pilot.start()
    asset_id = upload(args.release,bundle,os.environ["STREAM_CHECK_TOKEN"],commit)
    sending = corpus_pilot.timing(tick)
    receipt = {"run_id":run_id, "commit":commit, "worker":str(uuid4()), "nodes":args.nodes, "slot":args.slot,
               "plan_sha256":run.digest(PLAN_PATH), "release_id":args.release, "asset_id":asset_id,
               "archive_name":name, "archive_sha256":bundle_sha, "archive_bytes":bundle.stat().st_size,
               "document_ids":indices, "token_count":sum(part["tokens"] for part in rows.values()),
               "setup":setup, "compute":compute, "packing":packing, "upload":sending}
    publish((run_id,commit,TRACK,args.nodes,args.slot),receipt,title="Full model corpus bundle receipt")


if __name__ == "__main__":
    main()
