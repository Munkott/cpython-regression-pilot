"""Bounded client for this project's private draft assets and public receipts."""
import http.client
import json
import subprocess
import time
from urllib.parse import urlsplit
import run
from stream_pilot import REPO

ASSET_HOSTS={"release-assets.githubusercontent.com","objects.githubusercontent.com","github-releases.githubusercontent.com"}
now=run.utc

def journal(path,facts):
    with path.open("a") as out:
        print(json.dumps(facts,separators=(",",":")),file=out)


class GitHub:
    def __init__(self, log):
        self.auth=subprocess.check_output(["gh","auth","token","--hostname","github.com"],timeout=8,text=True).strip()
        self.log=log
        self.connection=None

    def request(self,path,method="GET",data=None):
        start=(time.monotonic(),run.cpu())
        try:
            if self.connection is None:
                self.connection=http.client.HTTPSConnection("api.github.com",timeout=15)
            headers={"Authorization":"Bearer "+self.auth,"User-Agent":"overhead-si-queue",
                     "Accept":"application/vnd.github+json","Content-Type":"application/json"}
            self.connection.request(method,"/repos/"+REPO+path,
                                    json.dumps(data).encode() if data is not None else None,headers)
            response=self.connection.getresponse()
            body=response.read(4_000_001)
            journal(self.log,{"method":method,"path":path,"utc":now(),"status":response.status,
                              "bytes":len(body),"cost":{"wall_s":time.monotonic()-start[0],"cpu_s":run.cpu()-start[1]}})
            if response.status not in (200,201,202,204) or len(body)>4_000_000:
                raise RuntimeError(f"GitHub {method} HTTP {response.status} on {path}")
            return json.loads(body) if body else None
        except (OSError,http.client.HTTPException):
            self.connection.close()
            self.connection=None
            raise

    def get(self,path):
        for attempt in range(3):
            try:
                return self.request(path)
            except (OSError,http.client.HTTPException):
                if attempt==2:
                    raise
                time.sleep(1)

def download(api, asset, target):
    connection=http.client.HTTPSConnection("api.github.com",timeout=20)
    headers={"Authorization":"Bearer "+api.auth,"Accept":"application/octet-stream","User-Agent":"overhead-si-weights"}
    try:
        connection.request("GET",f"/repos/{REPO}/releases/assets/{asset}",headers=headers)
        response=connection.getresponse()
        if response.status==302:
            destination=urlsplit(response.getheader("Location"))
            assert destination.scheme=="https" and destination.hostname in ASSET_HOSTS and not destination.username
            connection.close()
            connection=http.client.HTTPSConnection(destination.hostname,timeout=20)
            path=destination.path+("?" + destination.query if destination.query else "")
            connection.request("GET",path)
            response=connection.getresponse()
        if response.status!=200:
            raise OSError(f"download HTTP {response.status}")
        body=response.read(1_500_001)
        assert len(body)<=1_500_000
        target.write_bytes(body)
    finally:
        connection.close()

