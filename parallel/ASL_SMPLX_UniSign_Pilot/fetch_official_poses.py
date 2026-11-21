"""Fetch individual published How2Sign pose files without the full split archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen
import zipfile

REVISION = "eab251b7fe7e8521afc0e67be98add670ea40a0d"
ROOT = f"https://huggingface.co/ZechengLi19/Uni-Sign/resolve/{REVISION}"
LENGTHS = (2147483648, 2147483648, 2147483648, 2023023631)
TOTAL = sum(LENGTHS)


def fetch(part, first, end):
    url = f"{ROOT}/how2sign_pose_format.zip.{part:02d}"
    request = Request(url, headers={"Range": f"bytes={first}-{end - 1}"})
    expected = f"bytes {first}-{end - 1}/{LENGTHS[part]}"
    with urlopen(request, timeout=60) as response:
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise RuntimeError(f"Server did not honor range for archive part {part:02d}")
        data = response.read(end - first + 1)
    if len(data) != end - first:
        raise RuntimeError(f"Truncated range in archive part {part:02d}")
    return data


def read_range(first, end):
    data = []
    start = 0
    for part, size in enumerate(LENGTHS):
        left, right = max(start, first), min(start + size, end)
        if left < right:
            data.append(fetch(part, left - start, right - start))
        start += size
    if sum(map(len, data)) != end - first:
        raise ValueError("Archive offset is out of bounds")
    return b"".join(data)


def fetch_poses(ids, split, output):
    with tempfile.TemporaryFile() as archive:
        # Create a sparse view with just the ZIP64 directory and selected local members.
        tail_start = TOTAL - 8_000_000
        archive.seek(tail_start)
        archive.write(read_range(tail_start, TOTAL))
        with zipfile.ZipFile(archive) as zipped:
            names = {Path(item.filename).stem: item for item in zipped.infolist()
                     if item.filename.startswith(f"{split}-") and item.filename.endswith(".pkl")}
            if missing := set(ids) - names.keys():
                raise ValueError(f"Missing {split} samples: {', '.join(sorted(missing))}")
            directory_offsets = sorted(item.header_offset for item in zipped.infolist())
            next_offsets = dict(zip(directory_offsets, directory_offsets[1:] + [zipped.start_dir]))
            selected = [names[name] for name in ids]
            with ThreadPoolExecutor(max_workers=4) as workers:
                ranges = [(item.header_offset, next_offsets[item.header_offset]) for item in selected]
                # The final member can be far from the directory. Cap the fetch above
                # its compressed size by room for a local header and extra fields.
                ranges = [(start, min(end, start + item.compress_size + 131_200))
                          for (start, end), item in zip(ranges, selected)]
                futures = [workers.submit(read_range, start, end) for start, end in ranges]
                output.mkdir(parents=True, exist_ok=True)
                for item, future in zip(selected, futures):
                    archive.seek(item.header_offset)
                    archive.write(future.result())
                    archive.flush()
                    dest = output / Path(item.filename).name
                    dest.write_bytes(zipped.read(item))  # validates decompressed size and CRC
                    print(dest, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("ids", nargs="+")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in args.ids):
        parser.error("Use pose basenames without .pkl")
    fetch_poses(args.ids, args.split, args.output)


if __name__ == "__main__":
    main()
