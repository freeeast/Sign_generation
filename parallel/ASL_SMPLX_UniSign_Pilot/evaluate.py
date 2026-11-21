"""Decode each pose condition with the same frozen How2Sign Uni-Sign model."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from src.pose import load_pose
from src.unisign import UniSign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mt5', type=Path, required=True)
    parser.add_argument('--real-dir', type=Path)
    parser.add_argument('--generated-dir', type=Path)
    parser.add_argument('--label', default='rig_proxy')
    parser.add_argument('--expected', type=int, default=24)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.sample.read_text())
    assert len(records) == args.expected and len({r['id'] for r in records}) == len(records)
    paths = []
    for record in records:
        if args.real_dir:
            paths.append((record, 'real_rtmpose', args.real_dir / (record['clip_id'] + '.pkl')))
        if args.generated_dir:
            paths.append((record, args.label, args.generated_dir / (record['id'] + '.npz')))
    if not paths:
        parser.error('Specify at least one pose directory')
    for _, _, path in paths:
        if not path.is_file():
            parser.error(f'Missing input {path}')
    model = UniSign(args.repo, args.checkpoint, args.mt5, args.device, args.threads)
    modes = {'real_rtmpose', args.label}
    decoys = {r['id']: r2['reference'] for r, r2 in zip(records, records[1:] + records[:1])}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['id', 'clip_id', 'condition', 'input_frames',
            'prediction', 'reference_nll', 'decoy_nll', 'forced_choice_win', 'seconds'])
        writer.writeheader()
        for r, condition, path in paths:
            assert condition in modes
            xy, scores = load_pose(path, trusted_pickle=condition == 'real_rtmpose')
            targets = [r['reference'], decoys[r['id']]]
            t0 = time.monotonic()
            result = model.infer(xy, scores, targets, max_frames=256, beams=4, max_tokens=32)
            ref, decoy = result['target_nll']
            writer.writerow(dict(id=r['id'], clip_id=r['clip_id'], condition=condition,
                input_frames=result['frames'], prediction=result['prediction'],
                reference_nll=round(ref, 7), decoy_nll=round(decoy, 7),
                forced_choice_win=int(ref < decoy), seconds=round(time.monotonic()-t0, 3)))
            file.flush()
            print(r['id'], condition, round(ref, 3), round(decoy, 3), repr(result['prediction']), flush=True)


if __name__ == '__main__':
    main()
