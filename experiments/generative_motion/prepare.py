"""Fetch and verify the bounded NSA snapshot used by the historical data loader."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

from scripts.fetch_nsa_validation_prefix import URL

HASHES = Path("experiments/SharedMotion/analysis/nsa_inputs.sha256")
PREFIX_BYTES = 1_200_000_000
FIRST, LAST = 5_711_719_869, 6_426_140_848


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(output, url, first, last, checksum):
    count = last - first + 1
    name = output.name
    if output.exists():
        if output.stat().st_size != count or digest(output) != checksum:
            raise ValueError(f"Existing {output} differs from the pinned snapshot")
        print(f"{name}: verified", flush=True)
        return
    temporary = output.with_suffix(".part")
    try:
        request = Request(url, headers={"Range": f"bytes={first}-{last}"})
        with urlopen(request, timeout=60) as response, temporary.open("xb") as stream:
            content_range = response.headers.get("Content-Range", "")
            if first and not content_range.startswith(f"bytes {first}-{last}/"):
                raise ValueError("Upstream did not honor the train directory byte range")
            remaining = count
            while remaining:
                block = response.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"Truncated {name}")
                stream.write(block)
                remaining -= len(block)
        if digest(temporary) != checksum:
            raise ValueError(f"Snapshot SHA-256 mismatch: {name}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{name}: downloaded and verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("work/shared"))
    args = parser.parse_args()
    checksums = {Path(name).name: checksum for checksum, name in (line.split() for line in HASHES.read_text().splitlines())}
    args.output.mkdir(parents=True, exist_ok=True)
    fetch(args.output / "nsa_prefix.bin", URL, 0, PREFIX_BYTES - 1, checksums["nsa_prefix.bin"])
    train_url = URL.replace("/h?", "/h/train_poses.zip?", 1)
    fetch(args.output / "train_central.bin", train_url, FIRST, LAST, checksums["train_central.bin"])


if __name__ == "__main__":
    main()
