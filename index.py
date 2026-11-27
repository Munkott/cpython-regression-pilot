"""Build and verify a deterministic text index of CPython test sources."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest, sha1, sha256
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import tarfile
import time

IDENT = re.compile(rb"[A-Za-z_][A-Za-z_0-9]*")
SCOPE = "Lib/test/"
MODES = {"100644", "100755"}
PAIRS = {
    "small": ("5dbe2ac850b7b180685ed6008d5b9d459fd4934f", "937828f5194d4d839c68f946dcf416ca8b367735"),
    "large": ("60403a5409ff2c3f3b07dd2ca91a7a3e096839c7", "ebf955df7a89ed0c7968f79faec1de49f61ed7cb"),
}


def in_scope(path):
    return path.startswith(SCOPE) and path.endswith(".py")


def cpu():
    times = [resource.getrusage(kind) for kind in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(t.ru_utime + t.ru_stime for t in times)


def elapsed(start):
    return {"wall_s": time.perf_counter()-start[0], "cpu_s": cpu()-start[1]}


def now():
    return time.perf_counter(), cpu()


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode() + b"\n"


def write(path, value):
    Path(path).write_bytes(encode(value))


def digest(value):
    return sha256(encode(value)).hexdigest()


def git(repo, *args, input=None):
    return subprocess.check_output(["git", "-C", str(repo), *args], input=input)


def manifest(repo, commit):
    data = git(repo, "ls-tree", "-rlz", commit, "--", SCOPE)
    paths = {}
    blobs = {}
    for row in data.rstrip(b"\0").split(b"\0") if data else []:
        meta, raw = row.split(b"\t", 1)
        mode, kind, oid, size = meta.decode().split()
        path = raw.decode("utf-8")
        if mode in MODES and kind == "blob" and in_scope(path):
            paths[path] = oid
            blobs[oid] = int(size)
    return paths, blobs


def local_delta(repo, base, head, old):
    paths = old.copy()
    raw = git(repo, "diff-tree", "--no-renames", "--no-commit-id", "--abbrev=40",
              "-r", "-z", base, head, "--", SCOPE).split(b"\0")
    for i in range(0, len(raw)-1, 2):
        before, after, old_oid, new_oid, status = raw[i].decode().removeprefix(":").split()
        path = raw[i+1].decode()
        if not in_scope(path):
            continue
        if before in MODES and paths.pop(path, None) != old_oid:
            raise ValueError(f"Inconsistent parent object: {path}")
        if after in MODES:
            paths[path] = new_oid
    return paths


def api_tree(commit):
    url = f"https://api.github.com/repos/python/cpython/git/trees/{commit}?recursive=1"
    args = ["curl", "-4", "-L", "--proto", "=https", "--compressed", "-fsS",
            "--max-time", "35", "--max-filesize", "5000000", "-w", "\n%{size_download}", url]
    raw, wire = subprocess.check_output(args).rsplit(b"\n", 1)
    result = json.loads(raw)
    if result["truncated"]:
        raise ValueError(f"Incomplete GitHub tree: {commit}")
    paths = {}
    for entry in result["tree"]:
        path = entry["path"]
        if entry["mode"] in MODES and entry["type"] == "blob" and in_scope(path):
            if path in paths:
                raise ValueError(f"Duplicate tree path: {path}")
            paths[path] = entry["sha"]
    return paths, {"payload_bytes": len(raw), "wire_body_bytes": int(wire), "sha256": sha256(raw).hexdigest(),
                   "root_tree": result["sha"], "url": url}


def read_blobs(repo, oids):
    if not oids:
        return []
    request = "".join(oid + "\n" for oid in oids).encode()
    raw = git(repo, "cat-file", "--batch", input=request)
    rows = []
    pos = 0
    for oid in oids:
        end = raw.index(b"\n", pos)
        actual, mode, size = raw[pos:end].decode().split()
        if actual != oid or mode != "blob":
            raise ValueError(f"Unexpected Git object {actual}")
        start = end + 1
        pos = start + int(size) + 1
        rows.append((oid, raw[start:pos-1]))
    if pos != len(raw):
        raise ValueError("Unexpected trailing Git bytes")
    return rows


def token_record(oid, body):
    check = sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
    if check != oid:
        raise ValueError(f"Blob content mismatch: {oid}")
    return oid, (sha256(body).hexdigest(), sorted({word.decode() for word in IDENT.findall(body)}))


def compute_chunk(rows):
    return dict(token_record(oid, body) for oid, body in rows)


def weighted_bins(oids, sizes, workers):
    bins = [([], 0) for _ in range(workers)]
    for oid in sorted(oids, key=lambda o: (-sizes[o], o)):
        slot = min(range(workers), key=lambda i: bins[i][1])
        entries, size = bins[slot]
        entries.append(oid)
        bins[slot] = entries, size + sizes[oid]
    return [sorted(entries) for entries, _ in bins]


def collect(paths, records, commit):
    inverse = {}
    files = {}
    for path, oid in sorted(paths.items()):
        sha, terms = records[oid]
        files[path] = [oid, sha]
        for term in terms:
            inverse.setdefault(term, []).append(path)
    return {"commit": commit, "files": files, "postings": inverse}


def prior_cache(repo, commit):
    start = now()
    paths, sizes = manifest(repo, commit)
    cached = compute_chunk(read_blobs(repo, sorted(sizes)))
    product = collect(paths, cached, commit)
    return paths, cached, product, elapsed(start)


def reference(archive, commit):
    """Construct from complete archive without the incremental code or cache."""
    files = {}
    postings = {}
    with tarfile.open(archive, "r:gz") as stream:
        for entry in stream:
            if not entry.isfile() or not in_scope(entry.name):
                continue
            body = stream.extractfile(entry).read()
            match = set(re.findall(rb"[A-Za-z_][A-Za-z_0-9]*", body))
            index_sha = sha256(body).hexdigest()
            git_sha = sha1(b"blob " + str(len(body)).encode() + b"\x00" + body).hexdigest()
            files[entry.name] = [git_sha, index_sha]
            for word in match:
                postings.setdefault(word.decode(), set()).add(entry.name)
    return {"commit": commit, "files": files,
            "postings": {term: sorted(paths) for term, paths in postings.items()}}


def match(out, expected):
    fingerprint = sha256(out).hexdigest()
    if fingerprint != expected:
        raise ValueError("The resulting index differs from the full reference")
    return fingerprint


def merge_results(expected_bins, deliveries, prior):
    accepted = {}
    duplicates = 0
    for slot, records in deliveries:
        if set(records) != set(expected_bins[slot]):
            raise ValueError(f"Wrong blob set in shard {slot}")
        if slot in accepted:
            if accepted[slot] != records:
                raise ValueError(f"Conflicting duplicate shard {slot}")
            duplicates += 1
            continue
        accepted[slot] = records
    missing = sorted(set(range(len(expected_bins))) - accepted.keys())
    combined = prior.copy()
    for records in accepted.values():
        combined.update(records)
    return combined, missing, duplicates


def incremental(repo, paths, cached, head, workers=0):
    needed = sorted(set(paths.values()) - cached.keys())
    start = now()
    rows = read_blobs(repo, needed)
    reading = elapsed(start)
    start = now()
    if workers:
        payload = dict(rows)
        bins = weighted_bins(needed, {k: len(v) for k, v in rows}, workers)
        with ProcessPoolExecutor(workers) as pool:
            chunks = list(pool.map(compute_chunk, [[(o, payload[o]) for o in ids] for ids in bins]))
        all_records, missing, _ = merge_results(bins, enumerate(chunks), cached)
        if missing:
            raise ValueError("Local worker did not return all records")
    else:
        all_records = cached | compute_chunk(rows)
    computing = elapsed(start)
    start = now()
    final = collect(paths, all_records, head)
    return final, {"read": reading, "compute": computing, "merge": elapsed(start),
                   "new_blobs": len(needed), "raw_work_bytes": sum(len(b) for _, b in rows)}


def run_local(repo, out, online):
    out.mkdir(exist_ok=True)
    results = {}
    for load, (base, head) in PAIRS.items():
        start = now()
        old, cached, parent, setup = prior_cache(repo, base)
        assert encode(parent) == encode(reference(Path("refs") / (base + ".tar.gz"), base))
        start = now()
        target = reference(Path("refs") / (head + ".tar.gz"), head)
        target_bytes = encode(target)
        reference_cost = elapsed(start)
        target_sha = sha256(target_bytes).hexdigest()
        (out / f"{load}.index.json").write_bytes(target_bytes)
        direct_blobs = {path: pair[0] for path, pair in target["files"].items()}
        removed = old.keys() - direct_blobs.keys()
        assert all(removed.isdisjoint(paths) for paths in target["postings"].values())
        for term in ("Proxy", "WeakSet", "Pickler", "unittest", "PYTHON"):
            expect = list(target["postings"].get(term, []))
            bodies = read_blobs(repo, sorted(set(direct_blobs.values())))
            has_term = {oid for oid, body in bodies if term.encode() in IDENT.findall(body)}
            assert expect == sorted(path for path, oid in direct_blobs.items() if oid in has_term), term
        assert local_delta(repo, base, head, old) == direct_blobs
        row = {"base": base, "head": head, "paths": len(direct_blobs), "removed": len(removed),
               "created": len(direct_blobs.keys()-old.keys()),
               "modified": sum(old[p] != oid for p, oid in direct_blobs.items() if p in old),
               "parent_setup": setup, "independent_reference": reference_cost,
               "index_sha256": target_sha, "index_bytes": len(target_bytes), "trials": []}
        for run in range(3):
            for route in (["local", "api"] if run % 2 == 0 else ["api", "local"]) if online else ["local"]:
                start = now()
                if route == "local":
                    paths = local_delta(repo, base, head, old)
                    requests = []
                else:
                    paths, receipt = api_tree(head)
                    requests = [receipt]
                detection = elapsed(start)
                if paths != direct_blobs:
                    raise ValueError("Source manifest differs from independent target manifest")
                product, metrics = incremental(repo, paths, cached, head)
                fingerprint = match(encode(product), target_sha)
                row["trials"].append({"route": route, "run": run, "detect": detection, "sha256": fingerprint,
                                      "network": requests, "total": elapsed(start), **metrics})
        row["local_process_trials"] = []
        for workers in (1, 2, 4, 6, 6, 4, 2, 1):
            start = now()
            paths = local_delta(repo, base, head, old)
            product, metrics = incremental(repo, paths, cached, head, workers)
            fingerprint = match(encode(product), target_sha)
            row["local_process_trials"].append({"workers": workers, "total": elapsed(start),
                                                "sha256": fingerprint, **metrics})
        if not online:
            recovery_start = now()
            needed = sorted(set(direct_blobs.values()) - cached.keys())
            all_input = dict(read_blobs(repo, needed))
            bins = weighted_bins(needed, {k: len(v) for k, v in all_input.items()}, 2)
            first = compute_chunk([(oid, all_input[oid]) for oid in bins[0]])
            _, missing, duplicates = merge_results(bins, [(0, first), (0, first)], cached)
            assert missing == [1] and duplicates == 1
            fallback_start = now()
            fallback = compute_chunk(read_blobs(repo, bins[1]))
            complete, missing, duplicates = merge_results(bins, [(0, first), (0, first), (1, fallback)], cached)
            assert not missing and duplicates == 1
            fingerprint = match(encode(collect(direct_blobs, complete, head)), target_sha)
            row["recovery"] = {"missing_blobs": len(bins[1]), "duplicate_shards": duplicates,
                               "sha256": fingerprint, "extra_fallback": elapsed(fallback_start),
                               "total": elapsed(recovery_start)}
        results[load] = row
        print(load, {k: row[k] for k in ("paths", "created", "removed", "modified", "index_bytes")}, flush=True)
        write(out / "results.json", results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    run_local(args.repo, args.out, args.online)
