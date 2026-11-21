"""Join locked sample IDs to the original realigned How2Sign test transcripts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

SOURCE = 'https://huggingface.co/datasets/aipieces/How2Sign/resolve/a4fb69196612b063e602ab95b8d2099e5eda74fe/how2sign_realigned_test.csv'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('how2sign_csv',type=Path,help=SOURCE)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    locked=json.loads(Path(__file__).with_name('sample_ids.json').read_text())
    raw=args.how2sign_csv.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=locked['source_sha256']:
        p.error('CSV bytes differ from the pinned How2Sign test CSV')
    rows={r['SENTENCE_NAME']:r['SENTENCE'].strip() for r in csv.DictReader(
        raw.decode('utf-8-sig').splitlines(),delimiter='\t')}
    records=[dict(r,reference=rows[r['clip_id']]) for r in locked['clips']]
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
