"""Prepare the public corpus, fit on the same standard runner, and return the weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import corpus_pilot
import full_corpus
import run
from stream_pilot import REPO, publish
import train_model

TRACK = "model-fit"
BUNDLE = "trained-model.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("release", type=int)
    args = parser.parse_args()
    if (os.environ.get("GITHUB_REPOSITORY"),os.environ.get("GITHUB_EVENT_NAME")) != (REPO,"workflow_dispatch"):
        raise ValueError("requires the dedicated public benchmark workflow")
    args.root.mkdir()
    start = train_model.clock()
    cache = args.root / "inputs"
    cache.mkdir()
    numpy = train_model.PLAN["numpy"]
    inputs = {**full_corpus.PLAN["inputs"],"numpy.whl":{
        "urls":[numpy["url"]],"sha256":numpy["sha256"],"size":16527618}}
    corpus_pilot.fetch_inputs(cache,inputs)
    acquisition = train_model.elapsed(start)
    prep = args.root / "prepared"
    tick = train_model.clock()
    subprocess.run([sys.executable,"-B",str(Path(full_corpus.__file__)),
                    str(cache),str(prep)],check=True)
    prepare = train_model.elapsed(tick)
    output = args.root / "model"
    tick = train_model.clock()
    subprocess.run([sys.executable,"-B",str(Path(train_model.__file__)),"fit",str(prep/"dataset"),
                    str(cache/"numpy.whl"),str(output)],check=True)
    fitting = train_model.elapsed(tick)
    report = json.loads((output/"report.json").read_bytes())
    bundle = args.root / BUNDLE
    with ZipFile(bundle,"w",compression=ZIP_DEFLATED,compresslevel=3) as archive:
        for name in ("checkpoint.npz","report.json"):
            archive.write(output/name,name)
    commit = os.environ["GITHUB_SHA"]
    run_id = int(os.environ["GITHUB_RUN_ID"])
    tick = train_model.clock()
    asset = full_corpus.upload(args.release,bundle,os.environ["STREAM_CHECK_TOKEN"],commit)
    upload = train_model.elapsed(tick)
    receipt = {"run_id":run_id,"commit":commit,"worker":str(uuid4()),"release_id":args.release,
               "asset_id":asset,"archive_name":bundle.name,"archive_sha256":run.digest(bundle),
               "archive_bytes":bundle.stat().st_size, "checkpoint_sha256":report["checkpoint_sha256"],
               "plan_sha256":run.digest(train_model.PLAN_PATH),"dataset_sha256":report["dataset_sha256"],
               "acquisition":acquisition,"prepare":prepare,"fit":fitting,"upload":upload,
               "total":train_model.elapsed(start)}
    publish((run_id,commit,TRACK,1,0),receipt,title="Trained checkpoint receipt")


if __name__ == "__main__":
    main()
