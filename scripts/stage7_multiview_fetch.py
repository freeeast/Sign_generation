"""Recover exact public How2Sign side clips by bounded ZIP byte ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import scripts.fetch_how2sign_members as ranges

SIDE=("https://drive.usercontent.google.com/download"
      "?id=1mxL7kJPNUzJ6zoaqJyxF1Krnjo4F-eQG&export=download&confirm=t")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    ranges.URL=SIDE
    rows=json.loads(a.manifest.read_text())
    records=[]
    with tempfile.TemporaryFile() as archive:
        ranges.fetch(archive,"bytes=-1048576")
        with zipfile.ZipFile(archive) as z:
            for row in rows:
                name=row["clip_id"]
                stems={name.replace("rgb_front","rgb_side"),name.replace("rgb_front","rgb_side_left")}
                matches=[i for i in z.infolist() if Path(i.filename).stem in stems and i.filename.endswith(".mp4")]
                if len(matches)!=1:
                    related=[i.filename for i in z.infolist() if row["video_id"] in i.filename][:5]
                    records.append(dict(clip=name,status="unmatched",related=related))
                    continue
                item=matches[0]
                if item.compress_size>8_000_000:
                    records.append(dict(clip=name,status="over_budget",member=item.filename))
                    continue
                start=item.header_offset
                end=min(z.start_dir,start+item.compress_size+4096)
                ranges.fetch(archive,f"bytes={start}-{end-1}",start,end)
                out=a.output/f"{name}.mp4"
                with z.open(item) as source,out.open("wb") as target:
                    shutil.copyfileobj(source,target)
                records.append(dict(clip=name,status="downloaded",member=item.filename,
                                    sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
    (a.output/"inventory.json").write_text(json.dumps(records,indent=2)+"\n")
    print(records,flush=True)


if __name__=="__main__":
    main()
