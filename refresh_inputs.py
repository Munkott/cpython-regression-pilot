"""Download and verify the pinned public source and runtime."""
from pathlib import Path
import subprocess
import sys

from refresh_pilot import PLAN
from run import digest


def main(folder):
    for name, filename in (("source", "source.tar.xz"), ("runtime", "runtime.tar.zst")):
        asset = PLAN[name]
        path = folder / filename
        subprocess.run(["curl", "-4", "-L", "--fail", "--silent", "--show-error",
                        "--proto", "=https", "--connect-timeout", "12", "--max-time", "85",
                        "--retry", "1", "--retry-max-time", "110",
                        "--max-filesize", str(asset["max_bytes"]), "-o", str(path), asset["url"]],
                       check=True)
        if digest(path) != asset["sha256"]:
            raise ValueError(f"wrong {name} download digest")
    subprocess.run(["tar", "--zstd", "-xf", str(folder / "runtime.tar.zst"),
                    "-C", str(folder), "python/install"], check=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
