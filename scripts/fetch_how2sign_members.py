"""Extract selected members of the official How2Sign validation front ZIP via HTTP Range."""
import argparse
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen
import zipfile

URL = "https://dataverse.csuc.cat/api/access/datafile/13764"


def fetch(archive, header, expected_first=None, expected_end=None):
    request = Request(URL, headers={"Range": header, "Accept-Encoding": "identity"})
    with urlopen(request, timeout=60) as response:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or match is None:
            raise ValueError("The server did not return the requested byte range")
        first, last, total = map(int, match.groups())
        if expected_first is not None and (first, last + 1) != (expected_first, expected_end):
            raise ValueError("The server returned a different range")
        archive.seek(first)
        remaining = last - first + 1
        while remaining:
            data = response.read(min(remaining, 2 ** 20))
            if not data:
                raise EOFError("Truncated HTTP range")
            archive.write(data)
            remaining -= len(data)
    return total


def download(clips, output):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        fetch(archive, "bytes=-262144")
        with zipfile.ZipFile(archive) as zipped:
            for name in clips:
                items = [x for x in zipped.infolist() if Path(x.filename).name == name + ".mp4"]
                if len(items) != 1:
                    raise ValueError(f"{name}: expected one matching video, found {len(items)}")
                item = items[0]
                start = item.header_offset
                end = min(zipped.start_dir, start + item.compress_size + 131200)
                print(item.filename, item.file_size, flush=True)
                fetch(archive, f"bytes={start}-{end - 1}", start, end)
                path = output / (name + ".mp4")
                temporary = path.with_suffix(".partial")
                try:
                    with zipped.open(item) as source, temporary.open("wb") as target:
                        from shutil import copyfileobj
                        copyfileobj(source, target)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("video_names", nargs="+")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in args.video_names):
        parser.error("Use exact VIDEO_NAME without extension")
    download(args.video_names, args.output)
