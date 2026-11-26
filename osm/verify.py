"""Verify the recorded OSM results. Optionally reread archives, without altering the journal."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile
from collections import Counter

from observe import compare, inflate, MAX_UNCOMPRESSED, summary


def public_archive(base, seq):
    key = f'{seq:09}'
    url = base + '/'.join((key[:3], key[3:6], key[6:])) + '.osc.gz'
    with tempfile.TemporaryFile() as target:
        subprocess.run(['curl', '-fsS', '--max-time','20', '--max-filesize', str(8*1024*1024),
                        '-H','Accept-Encoding: identity',url], check=True, stdout=target)
        target.seek(0)
        return target.read()


def check(directory, online, stride=1):
    raw = (directory/'frozen.json').read_bytes()
    plan = json.loads(raw)
    assert hashlib.sha256((directory/'initial.state').read_bytes()).hexdigest() == plan['initial_state_sha256']
    assert plan['sequences'] == list(range(plan['anchor_sequence']+1,plan['anchor_sequence']+1+len(plan['sequences'])))
    archived = json.loads((directory/'results.json').read_bytes())
    db = sqlite3.connect((directory/'journal.sqlite').as_uri()+'?mode=ro', uri=True)
    assert db.execute("SELECT value FROM metadata WHERE key='plan_hash'").fetchone()[0] == hashlib.sha256(raw).hexdigest()
    assert json.loads(json.dumps(summary(db,plan))) == archived
    assert db.execute("PRAGMA integrity_check").fetchone() == ('ok',)
    rows=archived['rows']
    for seq, row in rows.items():
        assert row['matched'] == compare(row),seq
    segments = archived['exact_batches']
    listed = [seq for segment in segments for seq in segment['sequences']]
    assert listed == plan['sequences']
    for segment in segments:
        if 'bytes' in segment:
            independently_summed = sum(rows[str(seq)]['full']['answer']['size'] for seq in segment['sequences'])
            assert independently_summed == segment['bytes']
            assert independently_summed > plan['limit'] if segment['status']=='oversize' else independently_summed <= plan['limit']
    if archived['matched_count']==len(plan['sequences']):
        assert archived['exact_batches']==archived['range_batches_with_hindsight']
    first = gzip.compress(b'first',mtime=0)
    second = gzip.compress(b'second',mtime=0)
    control = inflate(first+second, {}, {'status':200,'code':0,'etag':None})
    assert control['members']==2 and control['size']==11 and (first+second)[-8:]==second[-8:]
    # Two bits suffice for a length/distance pair of at most 258 bytes.
    threshold=((1<<32)-1)//1032
    out={'journal_integrity':True,'no_duplicate_accepted_numbers':len(rows)==len(set(rows)),
         'reproducible_results':True,'two_member_counterexample':True,'compressed_bytes_bound_for_isize':threshold}
    if online:
        generators=Counter()
        selected=list(rows)[::stride]
        if rows and list(rows)[-1] not in selected:
            selected.append(list(rows)[-1])
        for seq in selected:
            row=rows[seq]
            raw_archive = public_archive(plan['archive_base'], int(seq))
            exact = row['full']['answer']
            assert hashlib.sha256(raw_archive).hexdigest()==exact['archive_sha256'],seq
            with gzip.GzipFile(fileobj=io.BytesIO(raw_archive)) as source:
                data=source.read(MAX_UNCOMPRESSED+1)
                assert len(data)==exact['size'] and not source.read(1),seq
            generator=re.search(rb'<osmChange\b[^>]*\bgenerator="([^"]+)"',data[:1024])
            generators[generator.group(1).decode() if generator else 'unavailable'] += 1
        out.update(online_full_archives=len(selected), independently_verified_sequences=[int(s) for s in selected], declared_generators=dict(generators))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--online',action='store_true')
    args=parser.parse_args()
    print(json.dumps(check(args.directory.resolve(),args.online),indent=2))
