"""Restore optional experiment assets from an owner-supplied migration ZIP."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import stat
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_BYTES = 486407969
ARCHIVE_SHA256 = "36f79943ed54effb7bfa5e3437cfa3bd51abcd375700dd0ed1b77cd45136c3b7"
ASSET_SUFFIXES = {".npz", ".npy", ".mp4", ".png", ".pdf"}
WEIGHT_DIRECTORY = PurePosixPath("experiments/SharedMotion/model")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prefix", action="append", default=[],
                        help="repository asset prefix to restore; repeatable; default all")
    options = parser.parse_args()
    archive = options.archive.resolve(strict=True)
    if archive.stat().st_size != ARCHIVE_BYTES or digest(archive) != ARCHIVE_SHA256:
        parser.error("Archive size or SHA-256 differs from the recorded migration bundle")
    restored = 0
    skipped = 0
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if member.is_dir():
                continue
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or name.parts[0] != "sign_motion_research":
                raise ValueError("Unexpected archive path")
            relative = Path(*name.parts[1:])
            is_shared_weight = (
                relative.suffix == ".pt"
                and PurePosixPath(relative.parent.as_posix()) == WEIGHT_DIRECTORY
            )
            if relative.suffix not in ASSET_SUFFIXES and not is_shared_weight:
                continue
            if options.prefix and not any(relative.as_posix().startswith(p) for p in options.prefix):
                continue
            target = (ROOT / relative).resolve()
            if not target.is_relative_to(ROOT) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Unsafe asset path or symlink")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".restore-tmp")
            try:
                with zipped.open(member) as source, temporary.open("xb") as output:
                    shutil.copyfileobj(source, output)
                if temporary.stat().st_size != member.file_size:
                    raise ValueError(f"Asset size mismatch: {relative}")
                if target.exists():
                    if target.is_file() and digest(target) == digest(temporary):
                        skipped += 1
                        continue
                    raise FileExistsError(f"Existing local asset differs: {target}")
                temporary.replace(target)
                restored += 1
            finally:
                temporary.unlink(missing_ok=True)
    print(f"Restored {restored} assets; verified {skipped} existing assets")


if __name__ == "__main__":
    main()
