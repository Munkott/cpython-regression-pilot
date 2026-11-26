"""Initialize a fresh minute-release window or verify a fixture on the receiving host."""
import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from observe import get, journal

STATE = 'https://planet.openstreetmap.org/replication/minute/state.txt'
ARCHIVES = 'https://osm-planet-eu-central-1.s3.dualstack.eu-central-1.amazonaws.com/planet/replication/minute/'


def prepare(target, fixture=None, *, cutoff=None):
    actual_source = ARCHIVES + 'state.txt'
    initial_error = None
    if fixture:
        body = fixture.read_bytes()
        transfer = None
    else:
        body, _, transfer = get(actual_source)
        if transfer['status'] != 200 or transfer['code']:
            initial_error = transfer
            actual_source = STATE
            body, _, transfer = get(actual_source)
            if transfer['status'] != 200 or transfer['code']:
                raise RuntimeError('could not read current published sequence from either official endpoint')
    if len(body) > 2048:
        raise ValueError('unexpected state size')
    values = dict(line.split('=', 1) for line in body.decode('ascii').splitlines()
                  if '=' in line and not line.startswith('#'))
    sequence = int(values['sequenceNumber'])
    stamp = datetime.fromisoformat(values['timestamp'].replace(chr(92), ''))
    now = datetime.now(timezone.utc)
    ceiling = 126 if cutoff else 10
    if cutoff is not None and cutoff <= now:
        raise ValueError('observation window already closed')
    if not 0 <= sequence <= 999_999_999-ceiling or stamp.tzinfo is None:
        raise ValueError('invalid published state')
    if not fixture and not 0 <= (now-stamp).total_seconds() <= 1800:
        raise ValueError('current public state is stale or ahead of the local clock')
    target.mkdir()
    (target / 'initial.state').write_bytes(body)
    plan = {
        'frozen_utc': now.isoformat(),
        'primary_deadline_utc': (cutoff or now + timedelta(minutes=35)).isoformat(),
        'archive_base': ARCHIVES,
        'state_source': actual_source,
        'initial_source_failure': initial_error,
        'anchor_sequence': sequence,
        'anchor_timestamp': stamp.isoformat(),
        'initial_state_sha256': sha256(body).hexdigest(),
        'sequences': list(range(sequence+1, sequence+1+ceiling)),
        'limit': 2097152,
        'initial_state_transfer': transfer,
        'fixture_only': fixture is not None,
    }
    content = (json.dumps(plan, indent=2) + '\n').encode()
    (target / 'frozen.json').write_bytes(content)
    database = journal(target / 'journal.sqlite', sha256(content).hexdigest())
    database.close()
    with sqlite3.connect((target / 'journal.sqlite').resolve().as_uri()+'?mode=ro', uri=True) as db:
        assert db.execute('SELECT COUNT(*) FROM receipts').fetchone() == (0,)
        assert db.execute('PRAGMA integrity_check').fetchone() == ('ok',)
    print(json.dumps({'folder': str(target), 'sequences': plan['sequences'],
                      'fixture_only': fixture is not None}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--cutoff')
    args = parser.parse_args()
    cutoff = datetime.fromisoformat(args.cutoff) if args.cutoff else None
    prepare(args.destination, args.fixture, cutoff=cutoff)
