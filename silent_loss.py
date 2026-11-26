"""Observe an expired training lease on a second hosted node and continue the shard."""
import argparse
import http.client
from multiprocessing import get_context
import os
from pathlib import Path
import signal
import time
from uuid import uuid4

import recovery_pilot as pilot
import run
from stream_pilot import REPO, digest, find, label, publish, request, unpack

MODEL=pilot.MODEL
PLAN=pilot.PLAN


def envelope(check,commit,name):
    blob=check["output"]["summary"]
    assert check["head_sha"]==commit and check["name"]==name and check["conclusion"]=="success"
    assert check["app"]["slug"]=="github-actions" and digest(blob)==check["external_id"]
    return unpack(blob)


def first(root,release,run_id,commit,owner,start):
    tasks=pilot.assign("fault-first")
    selection=pilot.prepare(root)
    prepared=run.utc()
    rows=pilot.execute("fault-first",tasks)
    receipt=pilot.emit(root,"fault-first",release,run_id,commit,owner,start,prepared,selection,None,tasks,rows)
    parent=pilot.checkpoint(run_id,commit,release)
    assert parent["id"]==receipt and parent["worker"]==owner
    task=pilot.choose_missing("fault-next",parent)[0]
    context=get_context("fork")
    active=context.Event()
    hold=context.Event()

    def probe():
        def did_first_step():
            active.set()
            if not hold.wait(100):
                raise RuntimeError("fault harness exceeded injection budget")
        pilot.worker.train(task,on_first_step=did_first_step)

    child=context.Process(target=probe)
    child.start()
    try:
        if not active.wait(25) or not child.is_alive():
            raise RuntimeError("could not observe first optimizer step")
        print(f"FIRST_OPTIMIZER_STEP task={task['i']} utc={run.utc()}",flush=True)
        for seq in (0,1):
            body=dict(run=run_id,commit=commit,release=release,plan_sha=pilot.worker.sweep.PLAN_SHA,
                      generation=1,worker=owner,checkpoint=receipt,sequence=seq,active=task["i"])
            publish((run_id,commit,"lease-pulse",2,seq),body,title="Training worker heartbeat")
            if seq==0:
                time.sleep(4)
        assert child.is_alive()
        os.kill(child.pid,signal.SIGKILL)
        child.join(10)
        assert child.exitcode==-signal.SIGKILL
        assert not (pilot.worker.MODELS/str(task["i"])/"checkpoint.npz").exists()
        print(f"SILENT_CHILD_KILLED task={task['i']} exit={child.exitcode} utc={run.utc()}",flush=True)
        # Keep the hosted job open long enough to test discovery without its status changing.
        time.sleep(48)
        print(f"SOURCE_JOB_STILL_LIVE utc={run.utc()}",flush=True)
    finally:
        if child.is_alive():
            child.kill()
            child.join(8)


def validate_pulse(check,run_id,commit,release,sequence):
    name=label(run_id,"lease-pulse",2,sequence)
    body=envelope(check,commit,name)
    assert (body["run"],body["commit"],body["release"],body["plan_sha"],body["generation"],body["sequence"])==(
        run_id,commit,release,pilot.worker.sweep.PLAN_SHA,1,sequence)
    return body


class Lease:
    def __init__(self,timeout):
        self.timeout=timeout
        self.sequence=-1
        self.last_seen=None

    def observe(self,seq,now):
        if seq>self.sequence:
            self.last_seen=now
            self.sequence=seq

    def expired(self,now):
        return self.sequence>=1 and now-self.last_seen>=self.timeout


def watchdog(root,release,run_id,commit,owner,start):
    selection=pilot.prepare(root)
    prepared=run.utc()
    lease=Lease(PLAN["lease_s"])
    latest=None
    for _ in range(160):
        try:
            checks=request("GET",f"/commits/{commit}/check-runs?filter=all&per_page=50")["check_runs"]
        except (OSError,http.client.HTTPException):
            time.sleep(PLAN["poll_s"])
            continue
        pulses={}
        for check in checks:
            for seq in (0,1):
                if check["name"]==label(run_id,"lease-pulse",2,seq):
                    assert seq not in pulses
                    pulses[seq]=(check,validate_pulse(check,run_id,commit,release,seq))
        tick=time.monotonic()
        if pulses:
            seq=max(pulses)
            lease.observe(seq,tick)
            latest=pulses[seq]
        if lease.expired(tick):
            assert len(pulses)==2 and all(pulses[0][1][name]==pulses[1][1][name]
                                          for name in ("worker","checkpoint","active"))
            break
        time.sleep(PLAN["poll_s"])
    else:
        raise RuntimeError("training lease never became eligible for recovery")
    pulse,body=latest
    parent=pilot.checkpoint(run_id,commit,release)
    assert selection==parent["selection_samples"] and body["checkpoint"]==parent["id"] and body["worker"]==parent["worker"]
    tasks=pilot.choose_missing("fault-next",parent)
    assert body["active"]==tasks[0]["i"]
    membership=[task["i"] for task in tasks]
    data=dict(run=run_id,commit=commit,release=release,plan_sha=pilot.worker.sweep.PLAN_SHA,generation=2,
              previous_generation=1,worker=owner,previous_worker=parent["worker"],checkpoint=parent["id"],
              pulse_id=pulse["id"],pulse_digest=pulse["external_id"],lease_s=PLAN["lease_s"],
              elapsed_without_pulse_s=time.monotonic()-lease.last_seen,pending=membership,detected=run.utc())
    grant=publish((run_id,commit,"lease-grant",2,0),data,title="Training lease expired and recovery granted")
    print(f"LEASE_EXPIRED grant={grant} last_pulse={pulse['id']}",flush=True)
    results=pilot.execute("fault-next",tasks)
    pilot.emit(root,"fault-next",release,run_id,commit,owner,start,prepared,selection,
               parent,tasks,results,grant=grant)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("mode",choices=("first","watchdog"))
    parser.add_argument("release",type=int)
    args=parser.parse_args()
    assert (os.environ["GITHUB_REPOSITORY"],os.environ["GITHUB_EVENT_NAME"],os.environ["GITHUB_RUN_ATTEMPT"])==(
        REPO,"workflow_dispatch","1")
    args.root.mkdir()
    action=first if args.mode=="first" else watchdog
    action(args.root,args.release,int(os.environ["GITHUB_RUN_ID"]),os.environ["GITHUB_SHA"],str(uuid4()),MODEL.clock())


if __name__=="__main__":
    main()
