"""Read a capped prefix of the official NSA download and recover complete validation fits."""
import argparse
from collections import defaultdict
from pathlib import Path, PurePosixPath
import re
import struct
from urllib.request import Request, urlopen
import zipfile

URL = "https://www.dropbox.com/scl/fo/nc4qxrwqz1ze2b4dy5wce/h?rlkey=zyx550j0lx88kj7wx6nfp3wjr&dl=1"
HEADER = struct.Struct("<IHHHHHIIIHH")
SIGNATURE = 0x04034B50


def local_members(data, offset):
    while offset + HEADER.size <= len(data):
        sig, _, flags, method, _, _, crc, packed, size, name_len, extra_len = HEADER.unpack_from(data, offset)
        if sig != SIGNATURE:
            return
        start = offset + HEADER.size
        name = data[start:start + name_len].decode()
        start += name_len + extra_len
        if flags & 8 and not packed:
            if name == "val_poses.zip" and method == 0:
                yield name, start, start, crc, size, method
                return
            raise ValueError(f"Unknown length for local member {name}")
        end = start + packed
        if end > len(data):
            return
        yield name, start, end, crc, size, method
        offset = end


def fetch(output, max_bytes, wanted):
    import zlib

    request = Request(URL, headers={"Range": f"bytes=0-{max_bytes - 1}", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=35) as response:
        data = response.read(max_bytes)  # Server may reply 200 with the entire 7.25 GB archive; stop here.
    base = next((start for name, start, *_ in local_members(data, 0) if name == "val_poses.zip"), None)
    if base is None:
        raise ValueError("Validation ZIP is unavailable in the download prefix")
    runs = defaultdict(list)
    started = []
    for name, start, end, crc, size, method in local_members(data, base):
        path = PurePosixPath(name)
        if len(path.parts) == 2 and name.endswith("/") and path.parts[0] == "poses":
            started.append(path.name)
        if len(path.parts) == 3 and path.parts[0] == "poses":
            clip = path.parts[1]
            match = re.fullmatch(rf"{re.escape(clip)}_(\d+)_3D\.pkl", path.name)
            if match:
                runs[clip].append((int(match[1]), path.name, start, end, crc, size, method))
    if len(started) < 2:
        raise ValueError("Prefix did not cover a whole validation clip")
    complete = {clip: sorted(runs[clip]) for clip in started[:-1] if runs[clip]}
    for clip, rows in complete.items():
        if [r[0] for r in rows] != list(range(len(rows))):
            raise ValueError(f"Non-contiguous fit frames: {clip}")
        print(clip, len(rows), flush=True)
    if not wanted <= complete.keys():
        raise ValueError(f"Requested clips outside completed prefix: {sorted(wanted - complete.keys())}")
    output.mkdir(parents=True, exist_ok=True)
    for clip in wanted:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", clip):
            raise ValueError(f"Invalid clip name: {clip}")
        with zipfile.ZipFile(output / (clip + ".zip"), "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for _, name, start, end, crc, size, method in complete[clip]:
                if method != 8 or size > 65536:
                    raise ValueError(f"Unexpected fit frame storage: {name}")
                decompressor = zlib.decompressobj(-15)
                frame = decompressor.decompress(data[start:end], 65537)
                if not decompressor.eof or len(frame) != size or zlib.crc32(frame) != crc:
                    raise ValueError(f"Fit frame failed CRC: {name}")
                archive.writestr(f"{clip}/{name}", frame)
    print(f"Read {len(data)} archive bytes; {len(complete)} contiguous clips ended in the prefix.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-bytes", type=int, default=8_000_000)
    parser.add_argument("clips", nargs="*")
    args = parser.parse_args()
    if not 262144 <= args.max_bytes <= 100_000_000:
        parser.error("Use a capped prefix between 256 KiB and 100 MB")
    fetch(args.output, args.max_bytes, set(args.clips))
