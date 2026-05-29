#!/usr/bin/env python
"""
Convert raw event NPZ files to per-frame voxel grids consumed by the
DSERT-RoLL training/evaluation pipeline.

Layout
------
Input  : <data-root>/<weather>/<sequence>/rectified_EVENT_<L|R>/*.npz
         Each NPZ holds a single key `event` with shape (N, 4), dtype
         uint32, columns [x, y, polarity, timestamp].

Output : <data-root>/<weather>/<sequence>/VOXEL_<L|R>/*.npz
         Each NPZ holds a single key `voxel` with shape
         (num_bins, height, width), dtype float32.

The released checkpoint was trained with the events from the last
1 / num_bins of each capture window, voxelized with bilinear time
interpolation. Run with the defaults (num_bins=5, 1152x704, side=L) to
match the released model. Pass --side both if you plan to train stereo
event variants.

Example
-------
    python detection/tools/preprocess_event_voxel.py \\
        --data-root detection/data/dsert-roll/processed_data \\
        --side L --workers 8
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm


def events_to_voxel_grid(events: np.ndarray, num_bins: int,
                          width: int, height: int) -> np.ndarray:
    """
    Bilinear-time voxelization of a (N, 4) event array
    laid out as [timestamp, x, y, polarity].
    """
    assert events.shape[1] == 4
    voxel = np.zeros((num_bins, height, width), np.float32).ravel()

    first_stamp = events[0, 0]
    last_stamp = events[-1, 0]
    deltaT = last_stamp - first_stamp
    if deltaT == 0:
        deltaT = 1.0

    events = events.astype(np.float64, copy=True)
    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int64)
    ys = events[:, 2].astype(np.int64)
    pols = events[:, 3].copy()
    pols[pols == 0] = -1  # remap {0,1} -> {-1,+1}

    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)

    valid = in_bounds & (tis >= 0) & (tis < num_bins)
    np.add.at(
        voxel,
        xs[valid] + ys[valid] * width + tis[valid] * width * height,
        vals_left[valid],
    )
    valid = in_bounds & ((tis + 1) >= 0) & ((tis + 1) < num_bins)
    np.add.at(
        voxel,
        xs[valid] + ys[valid] * width + (tis[valid] + 1) * width * height,
        vals_right[valid],
    )

    return voxel.reshape(num_bins, height, width)


def _convert_one_npz(event_path: Path, out_path: Path,
                     num_bins: int, width: int, height: int,
                     overwrite: bool) -> bool:
    if out_path.exists() and not overwrite:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)

    event_data = np.load(event_path)['event']
    if event_data.size == 0:
        voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    else:
        # raw layout: [x, y, polarity, timestamp]
        ev_t = event_data[:, 3]
        # Use the last (1 / num_bins) of the capture interval (training-time convention).
        cut = ev_t[-1] - (ev_t[-1] - ev_t[0]) / num_bins
        mask = ev_t > cut
        n = int(np.count_nonzero(mask))
        if n == 0:
            voxel = np.zeros((num_bins, height, width), dtype=np.float32)
        else:
            ev = np.empty((n, 4), dtype=np.float64)
            ev[:, 0] = event_data[mask, 3]  # timestamp
            ev[:, 1] = event_data[mask, 0]  # x
            ev[:, 2] = event_data[mask, 1]  # y
            ev[:, 3] = event_data[mask, 2]  # polarity
            voxel = events_to_voxel_grid(ev, num_bins, width, height)

    np.savez_compressed(out_path, voxel=voxel)
    return True


def _process_sequence(task) -> Tuple[str, int, int, str]:
    """Returns (seq_label, n_written, n_skipped, status)."""
    seq_dir, side, num_bins, width, height, overwrite = task
    label = f"{seq_dir.parent.name}/{seq_dir.name}"
    in_dir = seq_dir / f"rectified_EVENT_{side}"
    out_dir = seq_dir / f"VOXEL_{side}"
    if not in_dir.is_dir():
        return (label, 0, 0, "no rectified_EVENT_{} dir".format(side))
    files = sorted(in_dir.glob("*.npz"))
    if not files:
        return (label, 0, 0, "empty input dir")

    n_written = 0
    n_skipped = 0
    for f in files:
        try:
            if _convert_one_npz(f, out_dir / f.name, num_bins, width, height, overwrite):
                n_written += 1
            else:
                n_skipped += 1
        except Exception as e:  # noqa: BLE001
            return (label, n_written, n_skipped, f"error at {f.name}: {e}")
    return (label, n_written, n_skipped, "ok")


def _collect_sequences(data_root: Path) -> List[Path]:
    seqs: List[Path] = []
    for weather in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for seq in sorted(p for p in weather.iterdir() if p.is_dir()):
            seqs.append(seq)
    return seqs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--data-root', required=True, type=Path,
                    help='Path to processed_data/ (containing Clear/, Fog/, ...)')
    ap.add_argument('--side', choices=['L', 'R', 'both'], default='L',
                    help='Which event camera to process. '
                         'ours.yaml only consumes VOXEL_L, so the default is L.')
    ap.add_argument('--num-bins', type=int, default=5,
                    help='Temporal bins of the voxel grid (default: 5)')
    ap.add_argument('--width', type=int, default=1152, help='Voxel width  (default: 1152)')
    ap.add_argument('--height', type=int, default=704, help='Voxel height (default: 704)')
    ap.add_argument('--workers', type=int,
                    default=max(1, (os.cpu_count() or 4) // 2),
                    help='Parallel sequences to process (default: half of CPUs).')
    ap.add_argument('--overwrite', action='store_true',
                    help='Recompute voxel files even if they already exist.')
    args = ap.parse_args()

    if not args.data_root.is_dir():
        print(f"ERROR: --data-root {args.data_root} is not a directory", file=sys.stderr)
        return 1

    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    seqs = _collect_sequences(args.data_root)
    if not seqs:
        print(f"ERROR: no sequences found under {args.data_root}", file=sys.stderr)
        return 1

    print(f"Sequences : {len(seqs)} (under {args.data_root})")
    print(f"Sides     : {sides}")
    print(f"Voxel     : {args.num_bins} bins x {args.height} H x {args.width} W")
    print(f"Workers   : {args.workers}")
    print(f"Overwrite : {args.overwrite}")
    print("")

    tasks = [
        (s, side, args.num_bins, args.width, args.height, args.overwrite)
        for s in seqs for side in sides
    ]
    failed = []
    total_written = total_skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_process_sequence, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='sequences'):
            label, n_w, n_s, status = fut.result()
            total_written += n_w
            total_skipped += n_s
            if status != 'ok':
                failed.append((label, status))
                tqdm.write(f"[!] {label}: {status}")

    print("")
    print(f"Wrote {total_written} voxel files, skipped {total_skipped} existing.")
    if failed:
        print(f"{len(failed)} sequence(s) had problems:")
        for label, status in failed:
            print(f"  - {label}: {status}")
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
