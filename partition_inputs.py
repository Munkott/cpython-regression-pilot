"""Store separate synthetic input slices with a shared manifest and runtime."""
import json
from pathlib import Path
import tarfile

root = Path("docs")
truth = json.loads((root/"truth.json").read_bytes())
assert len(truth) == 2400
for slot in range(8):
    with tarfile.open(f"shard-{slot}.tar","w") as arc:
        for line in truth[slot::8]:
            path = root/line["file"]
            arc.add(path, arcname=path.name, recursive=False)
