"""Compare native gzip trailer answers and full archive reads, on a frozen OSM run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from resource import RUSAGE_CHILDREN, getrusage
import sqlite3
import struct
import subprocess
import tempfile
import time
import zlib
from email.utils import parsedate_to_datetime
from uuid import uuid4

MAX_TRANSFER = 8 * 1024 * 1024
MAX_UNCOMPRESSED = 64 * 1024 * 1024


def now():
    return datetime.now(timezone.utc)


def child_cpu():
    usage = getrusage(RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def get(path, ranged=False):
    start = time.monotonic()
    cpu = time.process_time() + child_cpu()
    with tempfile.TemporaryDirectory() as directory:
        body_path = Path(directory)/'body'
        header_path = Path(directory)/'headers'
        options = ['curl', '-sS', '--proto', '=https', '--max-time', '15', '--connect-timeout',
                   '7', '--max-filesize', str(MAX_TRANSFER), '-H', 'Accept-Encoding: identity',
                   '-D', str(header_path), '-o', str(body_path), '-w', '%{http_code} %{size_download}']
        if ranged:
            options += ['--range', '-8']
        result = subprocess.run([*options, path], capture_output=True, text=True, check=False)
        body = body_path.read_bytes() if body_path.is_file() else b''
        raw_headers = header_path.read_text(errors='replace') if header_path.is_file() else ''
        status, downloaded = result.stdout.split()
        headers = {}
        for line in raw_headers.splitlines():
            if line.startswith('HTTP/'):
                headers = {}
            elif ':' in line:
                key, value = line.split(':', 1)
                headers[key.lower()] = value.strip()
    receipt = {'method': 'range' if ranged else 'full', 'status': int(status),
               'code': result.returncode, 'downloaded_body_bytes': int(downloaded),
               'kept_body_bytes': len(body), 'cpu_s': time.process_time()+child_cpu()-cpu,
               'request_wall_s': time.monotonic()-start, 'received_at': now().isoformat(),
               'etag': headers.get('etag'), 'version': headers.get('x-amz-version-id'),
               'server_last_modified': headers.get('last-modified')}
    if result.returncode:
        receipt['error'] = result.stderr.strip()
    return body, headers, receipt


def read_range(body, headers, receipt):
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', headers.get('content-range', ''))
    if receipt['status'] != 206 or receipt['code'] or len(body) != 8 or not match:
        return None
    first, last, total = map(int, match.groups())
    if first != total-8 or last != total-1 or total < 18:
        return None
    crc, size = struct.unpack('<II', body)
    return {'crc32': crc, 'size': size, 'compressed': total}


def inflate(body, headers, receipt):
    if receipt['status'] != 200 or receipt['code'] or not body or (receipt['etag'] and headers.get('content-encoding')):
        return None
    cursor = body
    total = 0
    members = 0
    crc = 0
    while cursor:
        decoder = zlib.decompressobj(31)
        output = decoder.decompress(cursor, MAX_UNCOMPRESSED-total+1)
        total += len(output)
        if total > MAX_UNCOMPRESSED or not decoder.eof or decoder.unconsumed_tail:
            raise ValueError('archive is truncated or exceeds the declared output bound')
        members += 1
        crc = zlib.crc32(output)
        cursor = decoder.unused_data
        if cursor and not cursor.strip(b'\x00'):
            raise ValueError('archive includes trailing padding')
    return {'size': total, 'members': members, 'crc32_last_member': crc,
            'compressed': len(body), 'archive_sha256': hashlib.sha256(body).hexdigest()}


def fetch_parsed(url, ranged=False):
    body, headers, receipt = get(url, ranged)
    start = time.monotonic()
    cpu = time.process_time()
    try:
        info = (read_range if ranged else inflate)(body, headers, receipt)
    except (ValueError, zlib.error) as exc:
        receipt['validation_error'] = str(exc)
        info = None
    receipt['cpu_s'] += time.process_time()-cpu
    receipt['work_wall_s'] = receipt['request_wall_s'] + time.monotonic()-start
    receipt['answer'] = info
    return receipt


def state_info(base, sequence):
    data, _, receipt = get(base+'.state.txt')
    if receipt['status'] != 200 or receipt['code']:
        return None, receipt
    values = dict(line.split('=',1) for line in data.decode().splitlines() if '=' in line)
    if int(values['sequenceNumber']) != sequence:
        raise ValueError('state refers to another release')
    stamp = datetime.fromisoformat(values['timestamp'].replace(chr(92),''))
    mod = receipt['server_last_modified']
    return {'timestamp': stamp.isoformat(), 'last_modified': parsedate_to_datetime(mod).isoformat() if mod else None,
            'sha256': hashlib.sha256(data).hexdigest()}, receipt


def fetch_pair(url, first_range, seq, deadline, save):
    answers = {}
    for name in (['range','full'] if first_range else ['full','range']):
        for attempt in range(2):
            receipt = fetch_parsed(url, name=='range')
            answers[name] = receipt
            save(seq, name, receipt)
            if receipt['answer'] or now() >= deadline or receipt['status'] not in (0, 404, 500, 502, 503, 504):
                break
            time.sleep(2)
    return answers


def batches(rows, limit):
    filled = []
    active = []
    total = 0
    pending = False
    for seq, size in rows:
        if size is None or pending:
            pending = True
            if active:
                filled.append({'sequences':active,'bytes':total,'status':'open_before_gap'})
                active, total = [], 0
            filled.append({'sequences':[seq],'status':'awaiting_prior_size'})
            continue
        if active and total+size > limit:
            filled.append({'sequences':active,'bytes':total,'status':'batch'})
            active, total = [], 0
        if size > limit:
            filled.append({'sequences':[seq],'bytes':size,'status':'oversize'})
        else:
            active.append(seq)
            total += size
    if active:
        filled.append({'sequences':active,'bytes':total,'status':'batch'})
    return filled


def journal(path, plan_hash):
    db=sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS receipts (
            sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attempts (
            sequence INTEGER NOT NULL, route TEXT NOT NULL, payload TEXT NOT NULL
        );
    """)
    with db:
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('plan_hash', ?)", (plan_hash,))
    if db.execute("SELECT value FROM metadata WHERE key='plan_hash'").fetchone()[0] != plan_hash:
        raise ValueError('journal belongs to a different frozen plan')
    return db


def compare(pair):
    short, full = pair['range'], pair['full']
    small, exact = short['answer'], full['answer']
    etag = short['etag'] and short['etag']==full['etag']
    version = not (short['version'] or full['version']) or short['version']==full['version']
    return bool(small and exact and etag and version and exact['members']==1
                and small['crc32']==exact['crc32_last_member']
                and small['size']==exact['size'] and small['compressed']==exact['compressed'])


def summary(db, plan):
    rows = dict((seq,json.loads(payload)) for seq,payload in db.execute("SELECT * FROM receipts"))
    totals = {}
    for route in ('state','range','full'):
        r = [json.loads(row[0]) for row in db.execute("SELECT payload FROM attempts WHERE route=?", (route,))]
        totals[route] = {'requests':len(r), 'counts_by_status':{status:sum(e['status']==status for e in r) for status in sorted({e['status'] for e in r})},
                         **{name:sum(e.get(name,0) for e in r)
                            for name in ('cpu_s','request_wall_s','work_wall_s','downloaded_body_bytes')}}
    diagnostic = []
    reference = []
    planned = plan['sequences']
    frozen = datetime.fromisoformat(plan['frozen_utc'])
    for seq in planned:
        row = rows.get(seq, {})
        state = row.get('state')
        if not state:
            diagnostic.append((seq,None))
            reference.append((seq,None))
            continue
        row['data_window_after_freeze'] = datetime.fromisoformat(state['timestamp']) > frozen
        row['state_file_last_modified_after_freeze'] = bool(state['last_modified'] and datetime.fromisoformat(state['last_modified']) > frozen)
        row['matched'] = compare(row)
        baseline = row['full']['answer']
        diagnostic.append((seq,row['range']['answer']['size'] if row['matched'] else None))
        reference.append((seq,baseline['size'] if baseline else None))
    return {'rows':{k:rows[k] for k in planned if k in rows},
            'pending':[seq for seq in planned if seq not in rows],
            'matched_count':sum(row.get('matched',False) for row in rows.values()),
            'prospective_matched_count':sum(row.get('matched',False) and row['data_window_after_freeze'] for row in rows.values()),
            'range_batches_with_hindsight':batches(diagnostic,plan['limit']),
            'exact_batches':batches(reference,plan['limit']), 'totals':totals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--accepted-limit', type=int)
    args = parser.parse_args()
    if args.accepted_limit is not None and args.accepted_limit < 1:
        parser.error('accepted-limit must be positive')
    folder = args.directory
    raw = (folder/'frozen.json').read_bytes()
    plan = json.loads(raw)
    deadline = datetime.fromisoformat(plan['primary_deadline_utc'])
    db = journal(folder/'journal.sqlite', hashlib.sha256(raw).hexdigest())

    def save(seq, route, receipt):
        with db:
            db.execute("INSERT INTO attempts VALUES (?,?,?)", (seq,route,json.dumps(receipt)))

    def save_summary():
        data = summary(db,plan)
        staged = folder/'results.tmp'
        staged.write_text(json.dumps(data,indent=2)+'\n')
        staged.replace(folder/'results.json')

    if not args.summary_only:
        execution_id = uuid4().hex
        with (folder/'executions.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps({'execution': execution_id, 'started_at': now().isoformat()})+'\n')
        accepted = {seq for seq, payload in db.execute("SELECT sequence,payload FROM receipts")
                    if (row:=json.loads(payload)).get('accepted_at', True) and compare(row)}
        for idx, seq in enumerate(plan['sequences']):
            if args.accepted_limit is not None and len(accepted) >= args.accepted_limit:
                break
            if seq in accepted:
                continue
            digits = f'{seq:09}'
            base = plan['archive_base'] + '/'.join((digits[:3],digits[3:6],digits[6:]))
            while now() < deadline:
                state, receipt = state_info(base,seq)
                save(seq,'state',receipt)
                if state:
                    pair = fetch_pair(base+'.osc.gz',idx%2==0,seq,deadline,save)
                    matched = compare(pair)
                    stamp = now().isoformat() if matched else None
                    with db:
                        db.execute(
                            "INSERT INTO receipts VALUES (?,?) "
                            "ON CONFLICT(sequence) DO UPDATE SET payload=excluded.payload "
                            "WHERE json_extract(receipts.payload,'$.accepted_at') IS NULL",
                            (seq, json.dumps({'state': state, **pair, 'execution': execution_id,
                                              'accepted_at': stamp})),
                        )
                    save_summary()
                    print(seq,matched,[pair[v]['status'] for v in ('range','full')],flush=True)
                    if matched:
                        accepted.add(seq)
                        break
                if receipt['status'] not in (0,404,500,502,503,504):
                    break
                remaining = (deadline-now()).total_seconds()
                if remaining > 0:
                    time.sleep(min(15,remaining))
    save_summary()
    db.close()


if __name__=='__main__':
    main()
