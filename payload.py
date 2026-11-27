"""Group fixed queue decisions and prepare size-controlled checkpoint archives."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path
import random
import shutil

import run
from selector import Artifacts, scores
from warm_worker import COHORTS

SIZES = {"q144":None,"repeat144":4*1024**2,"q72":16*1024**2}


def padding(qid,slot,weight_size):
    target = SIZES[qid]
    size = 0 if target is None else target-weight_size
    if size<0:
        raise ValueError("Weight file exceeds the fixed transfer size")
    return random.Random(f"OSI transfer fixture v1 {qid} {slot}").randbytes(size)


def choose(worker):
    client=Artifacts()
    names=[f"scores-bulk-{qid}-{slot}" for qid in COHORTS for slot in range(8)]
    ids,stats=client.wait(names)
    with ThreadPoolExecutor(max_workers=6) as pool:
        packets=list(pool.map(lambda i:client.document(i,"receipt.json"),ids))
    selected={}
    for index,qid in enumerate(COHORTS):
        best=scores([value for value,_ in packets[8*index:8*index+8]],qid,client)
        selected[qid]={"candidate":best["i"],"slot":best["i"]%8,"sha256":best["sha"]}
    control=stats|{"downloaded_score_body_bytes":sum(size for _,size in packets)}
    run.put_json(worker/"decision.json",{"run":client.run_id,"commit":client.commit,"selection":selected,
                                        "created_at":run.utc(),"control":control})


def package(worker,mode,slot):
    client=Artifacts()
    if mode=="eager":
        chosen=COHORTS
    else:
        if slot==0:
            decision=json.loads((worker/"decision.json").read_bytes())
            traffic={"list_requests":0,"list_body_bytes":0,"wait_s":0,"decision_body_bytes":0}
        else:
            [aid],traffic=client.wait(["decision-bulk"],name_filter="decision-bulk")
            decision,traffic["decision_body_bytes"]=client.document(aid,"decision.json")
        assert (decision["run"],decision["commit"]) == (client.run_id,client.commit)
        assert set(decision["selection"])==set(COHORTS)
        chosen=[q for q,row in decision["selection"].items() if row["slot"]==slot]
        run.put_json(worker/"control.json",traffic|{"slot":slot,"winning_queues":chosen})
    for qid in chosen:
        source=worker/qid/"weights"/"checkpoint.npz"
        fingerprint=run.digest(source)
        if mode=="bulk":
            assert decision["selection"][qid]["sha256"]==fingerprint
        out=worker/qid/"payload"
        out.mkdir()
        shutil.copyfile(source,out/"checkpoint.npz")
        data=padding(qid,slot,source.stat().st_size)
        if data:
            (out/"padding.bin").write_bytes(data)
        run.put_json(out/"manifest.json",{"queue":qid,"slot":slot,
                                        "weight_bytes":source.stat().st_size,"weight_sha256":fingerprint,
                                        "padding_bytes":len(data),"padding_sha256":sha256(data).hexdigest()})
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        for qid in COHORTS:
            print(f"{qid}={'yes' if qid in chosen else 'no'}",file=output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("choose","eager","bulk"))
    p.add_argument("worker",type=Path)
    p.add_argument("slot",type=int)
    args=p.parse_args()
    if args.mode=="choose":
        assert args.slot==0
        choose(args.worker)
    else:
        package(args.worker,args.mode,args.slot)
