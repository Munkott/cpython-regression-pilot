"""Choose the winning slot from scored shards in one pinned workflow run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import ZipFile

import queue_run as queue
import run
import sweep
from transport import EXPECTED_REPOSITORY
from warm_worker import COHORTS


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if urlsplit(new_url).scheme != "https":
            raise ValueError("Expected an HTTPS artifact redirect")
        redirect = super().redirect_request(request,response,code,message,headers,new_url)
        if redirect is not None and urlsplit(new_url).netloc != urlsplit(request.full_url).netloc:
            redirect.remove_header("Authorization")
        return redirect


class Artifacts:
    def __init__(self):
        if os.environ.get("GITHUB_REPOSITORY") != EXPECTED_REPOSITORY or os.environ.get("GITHUB_EVENT_NAME") != "push":
            raise ValueError("Unexpected workflow source")
        self.run_id = int(os.environ["GITHUB_RUN_ID"])
        self.commit = os.environ["GITHUB_SHA"]
        self.base = "https://api.github.com/repos/" + EXPECTED_REPOSITORY
        self.token = os.environ["OSI_ACTIONS_TOKEN"]

    def get(self, path, limit):
        headers = {"Authorization": "Bearer "+self.token, "User-Agent":"osi-queue-benchmark",
                   "Accept":"application/vnd.github+json"}
        for attempt in range(4):
            try:
                with build_opener(SafeRedirect).open(Request(self.base+path,headers=headers),timeout=15) as response:
                    payload = response.read(limit+1)
                if len(payload)>limit:
                    raise ValueError("Artifact response too large")
                return payload
            except HTTPError as error:
                if error.code not in (404,429,500,502,503,504) or attempt==3:
                    raise RuntimeError(f"Artifact API returned HTTP {error.code}") from None
            except (TimeoutError,URLError):
                if attempt==3:
                    raise RuntimeError("Artifact API connection failed") from None
            time.sleep(1+attempt)

    def index(self,name=None):
        query = urlencode({"per_page":100,"name":name} if name else {"per_page":100})
        payload = self.get(f"/actions/runs/{self.run_id}/artifacts?{query}",200_000)
        result = json.loads(payload)
        if result["total_count"] > 100:
            raise ValueError("Artifact index unexpectedly needs pagination")
        return {r["name"]: r["id"] for r in result["artifacts"] if not r["expired"]},len(payload)

    def document(self, key, filename):
        raw = self.get(f"/actions/artifacts/{int(key)}/zip",100_000)
        with ZipFile(io.BytesIO(raw)) as archive:
            if archive.namelist()!=[filename] or archive.getinfo(filename).file_size>100_000:
                raise ValueError("Unexpected compressed artifact")
            doc = json.loads(archive.read(filename))
        return doc,len(raw)

    def wait(self, names, name_filter=None):
        start = time.monotonic()
        bytes_read = requests = 0
        while time.monotonic()-start < 150:
            index, size = self.index(name_filter)
            bytes_read += size
            requests += 1
            if all(name in index for name in names):
                return [index[name] for name in names], {"list_requests":requests,"list_body_bytes":bytes_read,
                                                        "wait_s":time.monotonic()-start}
            time.sleep(3)
        raise TimeoutError("Expected deliveries did not arrive")


def scores(rows, qid, client):
    config, = [c for c in queue.PLAN["cohorts"] if c["id"] == qid]
    tasks = {item["i"]:item for item in queue.tasks(config)}
    found = {}
    expected_samples = rows[0]["selection_samples"]
    for slot,receipt in enumerate(rows):
        assert (receipt["run"],receipt["commit"],receipt["queue"],receipt["slot"]) == (client.run_id,client.commit,qid,slot)
        assert receipt["plan_sha"] == sweep.PLAN_SHA
        assert receipt["selection_samples"] == expected_samples
        expected = {i for i in tasks if i%8==slot}
        assert expected == {r["i"] for r in receipt["rows"]} and len(expected)==len(receipt["rows"])
        for row in receipt["rows"]:
            assert all(row[k]==tasks[row["i"]][k] for k in ("lr","seed","train_seed"))
            assert all(math.isfinite(s) for s in row["scores"].values())
            found[row["i"]] = row
    assert found.keys()==tasks.keys()
    return min(found.values(),key=queue.winner_key)


def choose(client, mode, qid, worker):
    wanted = [f"scores-{mode}-{qid}-{slot}" for slot in range(8)]
    ids, metrics = client.wait(wanted)
    with ThreadPoolExecutor(max_workers=4) as pool:
        packets = list(pool.map(lambda aid:client.document(aid,"receipt.json"),ids))
    selected = scores([d for d,_ in packets],qid,client)
    decision = {"run":client.run_id,"commit":client.commit,"mode":mode,"queue":qid,
        "candidate":selected["i"],"slot":selected["i"]%8,"sha256":selected["sha"],"created_at":run.utc(),
        "control":{"downloaded_score_bytes":sum(size for _,size in packets),"files":len(packets)} | metrics}
    run.put_json(worker/qid/"decision.json",decision)


def admit(client, mode, qid, worker, slot):
    clock = time.monotonic()
    name = f"decision-{mode}-{qid}"
    selection = worker/qid/"decision.json"
    if slot==0:
        decision = json.loads(selection.read_bytes())
        traffic = {"list_requests":0,"list_body_bytes":0,"wait_s":0,"decision_body_bytes":0}
    else:
        [aid],traffic = client.wait([name],name_filter=name)
        decision,traffic["decision_body_bytes"] = client.document(aid,"decision.json")
    assert (decision["run"],decision["commit"],decision["mode"],decision["queue"]) == (client.run_id,client.commit,mode,qid)
    assert decision["candidate"]%8 == decision["slot"]
    publish = decision["slot"]==slot and mode!="fault"
    if publish:
        assert run.digest(worker/qid/"weights"/sweep.WEIGHT)==decision["sha256"]
    metrics = worker/"control"
    metrics.mkdir(exist_ok=True)
    run.put_json(metrics/f"{qid}.json", {"queue":qid,"slot":slot,"publish":publish,
                                       "elapsed_s":time.monotonic()-clock}|traffic)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as outputs:
        print("publish=" + ("yes" if publish else "no"),file=outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("choose","admit"))
    parser.add_argument("mode",choices=("selective","fault"))
    parser.add_argument("queue",choices=COHORTS)
    parser.add_argument("worker",type=Path)
    parser.add_argument("slot",type=int)
    opt = parser.parse_args()
    client = Artifacts()
    if opt.command=="choose":
        if opt.slot != 0:
            raise ValueError("Only the designated slot chooses a winner")
        choose(client,opt.mode,opt.queue,opt.worker)
    else:
        admit(client,opt.mode,opt.queue,opt.worker,opt.slot)
