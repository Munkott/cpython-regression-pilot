"""Fit a bounded queue of independent candidates next to the audited text corpus."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import corpus_pilot
import full_corpus
import run
from stream_pilot import REPO, publish
import train_model as model

FILE = Path(__file__).with_name("sweep_plan.json")
PLAN = json.loads(FILE.read_bytes())
PLAN_SHA = run.digest(FILE)
BASE = dict(model.PLAN)
WEIGHT = "checkpoint.npz"


def validation_examples(dataset, label):
    """Sample windows inside held-out documents, partitioned by source."""
    np = model.np
    corpus = model.load_corpus(dataset, "validation")
    manifest = json.loads((dataset/"manifest.json").read_bytes())
    docs = [row for row in manifest["documents"] if row["split"] == "validation"]
    # Documents in the training split and the other evaluation source stay out of this sample.
    indices = np.array([i for i,row in enumerate(docs) if row["source"]==PLAN[label]["source"]])
    assert len(indices) > 0
    lengths = (corpus.ends-corpus.previous)[indices]
    limits = np.cumsum(lengths)
    previous = np.concatenate(([0],limits[:-1]))
    draw = np.random.default_rng(PLAN[label]["seed"]).integers(0,limits[-1],
                             size=PLAN[label]["samples"],dtype=np.int64)
    part = np.searchsorted(limits,draw,side="right")
    positions = corpus.starts[indices[part]]+model.CTX+draw-previous[part]
    return *corpus.slices(positions), model.array_digest(positions)


def fit(task):
    variant, seed, rate = task["id"], task["seed"], task["learning_rate"]
    model.PLAN = dict(BASE, seed_initial=seed, learning_rate=rate)
    folder = OUTPUT/variant
    folder.mkdir()
    start = model.clock()
    report = model.fit(DATASET,folder)
    params = model.choose_parameters(folder/WEIGHT)
    x,y,sample_sha = SELECTION
    loss = model.evaluate(params,x,y)
    assert math.isfinite(loss)
    facts = {"variant":variant,"seed":seed,"learning_rate":rate,"selection_nll":loss,
        "selection_positions_sha256":sample_sha,"checkpoint_sha256":report["checkpoint_sha256"],
        "train_positions_sha256":report["train_positions_sha256"],"dataset_sha256":report["dataset_sha256"],
        "steps":report["steps"],"finished":run.utc(),"fit_time":model.elapsed(start)}
    run.put_json(folder/"receipt.json",facts)
    return facts


def setup_host(root):
    cache = root/"inputs"
    cache.mkdir()
    numpy = model.PLAN["numpy"]
    inputs = {**full_corpus.PLAN["inputs"],"numpy.whl":{
        "urls":[numpy["url"]],"sha256":numpy["sha256"],"size":16527618}}
    corpus_pilot.fetch_inputs(cache,inputs)
    prepared=root/"corpus"
    subprocess.run([sys.executable,"-B",full_corpus.__file__,str(cache),str(prepared)],check=True)
    return prepared/"dataset",cache/"numpy.whl"


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("--dataset",type=Path)
    cli.add_argument("--numpy",type=Path)
    cli.add_argument("--release",type=int)
    cli.add_argument("--nodes",type=int,choices=(1,2,4),default=1)
    cli.add_argument("--slot",type=int,default=0)
    args=cli.parse_args()
    hosted=args.release is not None
    root=args.root.resolve()
    root.mkdir()
    assert 0<=args.slot<args.nodes and (hosted or (args.nodes,args.slot)==(1,0))
    assert run.digest(model.PLAN_PATH)==PLAN["training_plan_sha256"]
    start=model.clock()
    if hosted:
        assert (os.environ["GITHUB_REPOSITORY"],os.environ["GITHUB_EVENT_NAME"])==(REPO,"workflow_dispatch")
        dataset,wheel=setup_host(root)
        run_id,commit=int(os.environ["GITHUB_RUN_ID"]),os.environ["GITHUB_SHA"]
    else:
        dataset,wheel=args.dataset,args.numpy
        run_id,commit=None,None
    model.load_numpy(wheel,root/"site")
    global OUTPUT, DATASET, SELECTION
    DATASET=dataset
    OUTPUT=root/"candidates"
    OUTPUT.mkdir()
    SELECTION=validation_examples(dataset,"selection")
    setup=model.elapsed(start)
    assigned=[task for i,task in enumerate(PLAN["variants"]) if i%args.nodes==args.slot]
    capacity=min(len(assigned),len(os.sched_getaffinity(0)))
    identity={"run_id":run_id,"commit":commit,"nodes":args.nodes,"slot":args.slot,
              "setup":setup,"prepared":run.utc(),"plan_sha256":PLAN_SHA,"capacity":capacity}
    compute=model.clock()
    results=[]
    with ProcessPoolExecutor(max_workers=capacity,mp_context=get_context("fork")) as pool:
        futures=[pool.submit(fit,task) for task in assigned]
        for future in as_completed(futures):
            facts=future.result()
            results.append(facts)
            if hosted:
                index=next(i for i,v in enumerate(PLAN["variants"]) if v["id"]==facts["variant"])
                publish((run_id,commit,"model-sweep-candidate",args.nodes,index),
                        dict(identity,candidate=facts),title="Model candidate validation receipt")
    best=min(results,key=lambda row:(row["selection_nll"],row["variant"]))
    summary=dict(identity,results=results,best=best["variant"],compute=model.elapsed(compute))
    if hosted:
        bundle=root/f"sweep-n{args.nodes}-slot{args.slot}.zip"
        with ZipFile(bundle,"w",compression=ZIP_DEFLATED,compresslevel=1) as archive:
            archive.write(OUTPUT/best["variant"]/WEIGHT,WEIGHT)
        assert bundle.stat().st_size < 1_500_000
        tick=model.clock()
        asset=full_corpus.upload(args.release,bundle,os.environ["STREAM_CHECK_TOKEN"],commit)
        proof=dict(identity,best=best["variant"],selection_nll=best["selection_nll"],asset_id=asset,
                   bundle=bundle.name,bundle_sha256=run.digest(bundle),bundle_bytes=bundle.stat().st_size,
                   checkpoint_sha256=best["checkpoint_sha256"],upload=model.elapsed(tick),total=model.elapsed(start))
        publish((run_id,commit,"model-sweep-winner",args.nodes,args.slot),proof,
                title="Best local candidate checkpoint receipt")
    run.put_json(root/"summary.json",dict(summary,total=model.elapsed(start)))


if __name__=="__main__":
    main()
