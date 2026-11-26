"""Move a committed OSM journal between disposable hosted workers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import full_corpus
from github_client import download
import run
from stream_pilot import REPO, digest, find, label, publish, unpack

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"osm"))
from verify import check as verify_osm

FILES={"frozen.json","initial.state","journal.sqlite","results.json","executions.jsonl"}
STAGES=("seed","resume","replay")
SIZES={"seed":3,"resume":10,"replay":10}
MAX_ARCHIVE=1_500_000
PINNED=tuple(sorted(ROOT/"osm"/name for name in ("observe.py","prepare.py","verify.py")))


def source_hash():
    return hashlib.sha256(b"".join(path.read_bytes() for path in PINNED)).hexdigest()


def utc():
    return datetime.now(timezone.utc)


def read(file):
    return json.loads(file.read_bytes())


def payloads(state):
    with sqlite3.connect(state/"journal.sqlite") as db:
        assert db.execute("PRAGMA integrity_check").fetchone()==("ok",)
        receipts=dict(db.execute("SELECT sequence,payload FROM receipts"))
        attempts=db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    return receipts,attempts


def verify_state(state):
    outcomes=verify_osm(state.resolve(),False)
    result=read(state/"results.json")
    receipts,attempts=payloads(state)
    assert result["matched_count"]==len(receipts)
    assert all(json.loads(row)["accepted_at"] for row in receipts.values())
    return outcomes,result,receipts,attempts


def replay_observe(state,limit=None):
    command=[sys.executable,"-B",str(ROOT/"osm/observe.py"),str(state)]
    if limit:
        command.extend(("--accepted-limit",str(limit)))
    subprocess.run(command,check=True)


def examine_bundle(archive,state):
    assert archive.stat().st_size<MAX_ARCHIVE
    with ZipFile(archive) as bundle:
        items=bundle.infolist()
        assert {i.filename for i in items}==FILES and len(items)==len(FILES)
        assert sum(i.file_size for i in items)<3_500_000
        state.mkdir()
        for name in FILES:
            (state/name).write_bytes(bundle.read(name))


def restore(root,stage,run_id,commit,release):
    tag="osm-journal-"+stage
    name=label(run_id,tag,1,0)
    for tries in range(4):
        item=find(commit,name)
        if item:
            break
        time.sleep(1+tries)
    else:
        raise RuntimeError("committed state receipt unavailable")
    blob=item["output"]["summary"]
    assert item["conclusion"]=="success" and item["app"]["slug"]=="github-actions"
    assert item["head_sha"]==commit and item["name"]==name
    assert item["external_id"]==digest(blob)
    facts=unpack(blob)
    assert (facts["run"],facts["commit"],facts["release"],facts["stage"])==(run_id,commit,release,stage)
    assert facts["source_hash"]==source_hash() and len(facts["accepted"])==SIZES[stage]
    token=os.environ["STREAM_CHECK_TOKEN"]
    detail=full_corpus.release_data(release,token)
    asset,=[asset for asset in detail["assets"] if asset["id"]==facts["asset_id"]]
    assert (asset["name"],asset["state"],asset["size"],asset["digest"])==(
        f"osm-{stage}.zip","uploaded",facts["bundle_bytes"],"sha256:"+facts["bundle_sha"])
    bundle=root/"restore.zip"
    download(SimpleNamespace(auth=token),asset["id"],bundle)
    assert bundle.stat().st_size==asset["size"] and run.digest(bundle)==facts["bundle_sha"]
    state=root/"live"
    examine_bundle(bundle,state)
    assert run.digest(state/"frozen.json")==facts["plan_sha"]
    assert run.digest(state/"journal.sqlite")==facts["journal_sha"]
    _,result,rows,count=verify_state(state)
    assert sorted(rows)==facts["accepted"] and count==facts["attempts"]
    assert result["matched_count"]==SIZES[stage]
    return state,{"id":item["id"],"digest":item["external_id"],"previous_worker":facts["worker"]}


def publish_state(root,state,stage,run_id,commit,release,parent,started,origin_cpu):
    _,result,rows,attempts=verify_state(state)
    plan=read(state/"frozen.json")
    assert len(rows)==SIZES[stage] and sorted(rows)==plan["sequences"][:len(rows)]
    bundle=root/f"osm-{stage}.zip"
    with ZipFile(bundle,"w",compression=ZIP_DEFLATED,compresslevel=3) as archive:
        for name in sorted(FILES):
            archive.write(state/name,name)
    assert bundle.stat().st_size<MAX_ARCHIVE
    asset_id=full_corpus.upload(release,bundle,os.environ["STREAM_CHECK_TOKEN"],commit)
    facts={"run":run_id,"commit":commit,"release":release,"stage":stage,"worker":str(uuid4()),"parent":parent,
           "asset_id":asset_id,"bundle_bytes":bundle.stat().st_size,"bundle_sha":run.digest(bundle),
           "journal_sha":run.digest(state/"journal.sqlite"),"plan_sha":run.digest(state/"frozen.json"),
           "source_hash":source_hash(),"accepted":sorted(rows),"attempts":attempts,
           "started":started.isoformat(),"saved":utc().isoformat(),
           "worker_cpu_s":run.cpu()-origin_cpu,
           "first_accepted":min(json.loads(row)["accepted_at"] for row in rows.values()),
           "last_accepted":max(json.loads(row)["accepted_at"] for row in rows.values())}
    publish((run_id,commit,"osm-journal-"+stage,1,0),facts,title="Verified OSM journal snapshot")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("stage",choices=STAGES)
    cli.add_argument("release",type=int)
    cli.add_argument("not_before")
    args=cli.parse_args()
    assert (os.environ["GITHUB_REPOSITORY"],os.environ["GITHUB_EVENT_NAME"],os.environ["GITHUB_RUN_ATTEMPT"])==(
        REPO,"workflow_dispatch","1")
    run_id,commit=int(os.environ["GITHUB_RUN_ID"]),os.environ["GITHUB_SHA"]
    started,origin_cpu=utc(),run.cpu()
    args.root.mkdir()
    parent=None
    if args.stage=="seed":
        threshold=datetime.fromisoformat(args.not_before)
        pause=(threshold-utc()).total_seconds()
        if not -300<=pause<=120:
            raise ValueError("activation time outside the experiment window")
        if pause>0:
            time.sleep(pause)
        state=args.root/"live"
        subprocess.run([sys.executable,"-B",str(ROOT/"osm/prepare.py"),str(state)],check=True)
        replay_observe(state,SIZES["seed"])
    else:
        previous=STAGES[STAGES.index(args.stage)-1]
        state,parent=restore(args.root,previous,run_id,commit,args.release)
        old,attempts=payloads(state)
        replay_observe(state)
        new,after=payloads(state)
        assert all(new[key]==value for key,value in old.items())
        if args.stage=="replay":
            assert old==new and attempts==after
            print(json.dumps(verify_osm(state.resolve(),True)),flush=True)
    publish_state(args.root,state,args.stage,run_id,commit,args.release,parent,started,origin_cpu)


if __name__=="__main__":
    main()
