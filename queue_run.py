"""Run one pinned candidate queue. Each node warms its environment once."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import run
from stream_pilot import REPO, publish
import sweep

MODEL=sweep.model
PLAN=sweep.PLAN
DATASET_PATH=Path("/tmp/overhead-corpus-scale.QZ0Ld0EZ/local/dataset")
WHEEL_PATH=Path("/tmp/overhead-model.8r60YcNj/numpy.whl")


def tasks(cohort):
    number=next(i for i,c in enumerate(PLAN["cohorts"]) if c["id"]==cohort["id"])
    return [{"i":i,"lr":PLAN["rates"][i%6],"seed":20261200+1000*number+i//6,
             "train_seed":20261300+number} for i in range(cohort["size"])]


def excluded(dataset):
    docs=json.loads((dataset/"manifest.json").read_bytes())["documents"]
    held=tuple(d["document"] for d in docs if d["split"]=="train" and
       (d["document"]==PLAN["holdout"]["gutenberg"] or PLAN["holdout"]["europarl_date"] in d["document"]))
    assert len(held)==12
    return held


def sample(dataset, split, source, filter_docs, seed, count):
    np=MODEL.np
    corpus=MODEL.load_corpus(dataset,split)
    docs=[d for d in json.loads((dataset/"manifest.json").read_bytes())["documents"] if d["split"]==split]
    chosen=np.array([i for i,d in enumerate(docs) if d["source"]==source and
                     (filter_docs is None or d["document"] in filter_docs)])
    assert len(chosen)>0
    sizes=(corpus.ends-corpus.previous)[chosen]
    limits=np.cumsum(sizes)
    offsets=np.concatenate(([0],limits[:-1]))
    draw=np.random.default_rng(seed).integers(0,limits[-1],size=count,dtype=np.int64)
    idx=np.searchsorted(limits,draw,side="right")
    positions=corpus.starts[chosen[idx]]+MODEL.CTX+draw-offsets[idx]
    x,y=corpus.slices(positions)
    return x,y,MODEL.array_digest(positions)


def examples(dataset,mode):
    section=PLAN[mode]
    split,filt=("validation",None) if mode=="selection" else ("train",excluded(dataset))
    return {key:sample(dataset,split,source,filt,section["seed"]+i,section["samples_each"])
            for i,(key,source) in enumerate((("books","gutenberg-api.zip"),("parliament","europarl-raw.zip")))}


def scores(parameters, samples):
    return {key:MODEL.evaluate(parameters,x,y) for key,(x,y,_) in samples.items()}


def winner_key(row):
    return sum(row["scores"].values()),row["i"]


def train(task):
    MODEL.PLAN=dict(sweep.BASE,seed_initial=task["seed"],learning_rate=task["lr"],seed_train=task["train_seed"])
    folder=MODELS/str(task["i"])
    folder.mkdir()
    start=MODEL.clock()
    report=MODEL.fit(DATASET,folder,exclude_documents=EXCLUDED)
    params=MODEL.choose_parameters(folder/sweep.WEIGHT)
    row=dict(task,scores=scores(params,SELECTION),sha=report["checkpoint_sha256"],
             train_sha=report["train_positions_sha256"],wall=MODEL.elapsed(start),finished=run.utc())
    assert all(map(math.isfinite,row["scores"].values()))
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("queue")
    parser.add_argument("--nodes",type=int,choices=(1,2,4),default=1)
    parser.add_argument("--slot",type=int,default=0)
    parser.add_argument("--release",type=int)
    parser.add_argument("--dataset",type=Path,default=DATASET_PATH)
    parser.add_argument("--wheel",type=Path,default=WHEEL_PATH)
    opt=parser.parse_args()
    hosted=opt.release is not None
    root=opt.root.resolve()
    root.mkdir()
    cohort=next(c for c in PLAN["cohorts"] if c["id"]==opt.queue)
    members=[t for t in tasks(cohort) if t["i"]%opt.nodes==opt.slot]
    start=MODEL.clock()
    assert run.digest(MODEL.PLAN_PATH)==PLAN["training_plan_sha256"]
    if hosted:
        assert (os.environ["GITHUB_REPOSITORY"],os.environ["GITHUB_EVENT_NAME"])==(REPO,"workflow_dispatch")
        dataset,wheel=sweep.setup_host(root)
        run_id,commit=int(os.environ["GITHUB_RUN_ID"]),os.environ["GITHUB_SHA"]
    else:
        dataset,wheel=opt.dataset,opt.wheel
        run_id,commit=None,None
    MODEL.load_numpy(wheel,root/"site")
    global MODELS, DATASET, EXCLUDED, SELECTION
    DATASET=dataset
    EXCLUDED=excluded(dataset)
    SELECTION=examples(dataset,"selection")
    MODELS=root/"models"
    MODELS.mkdir()
    capacity=min(len(members),len(os.sched_getaffinity(0)))
    facts=dict(queue=cohort["id"],nodes=opt.nodes,slot=opt.slot,capacity=capacity,start=run.utc(),
               setup=MODEL.elapsed(start),plan_sha=sweep.PLAN_SHA,release=opt.release,
               run=run_id,commit=commit,selection_samples={k:v[2] for k,v in SELECTION.items()})
    rows,pending=[],[]
    packet_size=PLAN["worker_publication_packet"]
    packet_number=0
    with ProcessPoolExecutor(max_workers=capacity,mp_context=get_context("fork")) as pool:
        jobs=[pool.submit(train,t) for t in members]
        for job in as_completed(jobs):
            row=job.result()
            rows.append(row)
            pending.append(row)
            if not hosted:
                with (root/"progress.jsonl").open("a") as journal:
                    print(json.dumps(row,separators=(",",":")),file=journal)
            if hosted and len(pending)==packet_size:
                publish((run_id,commit,"queue-packet",opt.nodes,100*opt.slot+packet_number),
                        dict(facts,rows=pending),title="Training queue results")
                packet_number+=1
                pending.clear()
        if hosted and pending:
            publish((run_id,commit,"queue-packet",opt.nodes,100*opt.slot+packet_number),
                    dict(facts,rows=pending),title="Training queue results")
    best=min(rows,key=winner_key)
    weight=MODELS/str(best["i"])/sweep.WEIGHT
    if hosted:
        asset=root/f"{cohort['id']}-n{opt.nodes}-s{opt.slot}.zip"
        with ZipFile(asset,"w",compression=ZIP_DEFLATED,compresslevel=1) as target:
            target.write(weight,sweep.WEIGHT)
        assert asset.stat().st_size<1_500_000
        asset_id=sweep.full_corpus.upload(opt.release,asset,os.environ["STREAM_CHECK_TOKEN"],commit)
        result=dict(facts,best=best["i"],asset=asset_id,bundle=asset.name,
                    sha=run.digest(asset),weight_sha=best["sha"],bytes=asset.stat().st_size,total=MODEL.elapsed(start))
        publish((run_id,commit,"queue-winner",opt.nodes,opt.slot),result,title="Training queue best weights")
        held=None
    else:
        held=scores(MODEL.choose_parameters(weight),examples(dataset,"holdout"))
    run.put_json(root/"result.json",dict(facts,rows=rows,best=best["i"],holdout=held,total=MODEL.elapsed(start)))


if __name__=="__main__":
    main()
