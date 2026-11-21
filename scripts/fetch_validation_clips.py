"""Fetch exact original-timing clips by SENTENCE_NAME from a How2Sign video mirror."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path

import requests

DATASET = "Flimdejong/how2sign"
REVISION = "8ebd757f151ab3cb74741d961abf93f0e36fff40"


def download(annotations, output, clips):
    with annotations.open(newline="", encoding="utf-8-sig") as stream:
        rows = {row["SENTENCE_NAME"]: i for i, row in enumerate(csv.DictReader(stream, delimiter="\t"))}
    output.mkdir(parents=True, exist_ok=True)

    def one(clip):
        response = requests.get("https://datasets-server.huggingface.co/rows", params={
            "dataset": DATASET, "config": "default", "split": "val",
            "offset": max(0, min(rows[clip] - 5, 1728)), "length": 11,
        }, timeout=60)
        response.raise_for_status()
        records = [r["row"] for r in response.json()["rows"] if r["row"]["sentence_name"] == clip]
        if len(records) != 1:
            raise ValueError(f"Could not resolve one exact video for {clip}")
        url = records[0]["video"]["src"]
        if f"/{REVISION}/" not in url:
            raise ValueError(f"Mirror revision changed for {clip}")
        image = requests.get(url, timeout=60)
        image.raise_for_status()
        if image.content[4:8] != b"ftyp":
            raise ValueError(f"Expected an MP4 for {clip}")
        path = output / (clip + ".mp4")
        path.write_bytes(image.content)
        print(clip, len(image.content), flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, clips))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("clips", nargs="+")
    args = parser.parse_args()
    download(args.annotations, args.output, args.clips)
