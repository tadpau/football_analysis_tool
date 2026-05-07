"""Inspect a tracker stub to debug detection issues.

Usage:
    python scripts/inspect_stub.py stubs/clip5.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np


def main(stub_path: str) -> None:
    with open(stub_path, "rb") as f:
        tracks = pickle.load(f)

    n = len(tracks)
    print(f"Loaded {n} frames from {stub_path}\n")

    # --- Class counts -----------------------------------------------------
    counts = {"player": 0, "goalkeeper": 0, "referee": 0, "ball": 0}
    ball_present = 0
    for ft in tracks:
        for k in counts:
            counts[k] += len(ft.get(k, {}))
        if ft.get("ball"):
            ball_present += 1

    print("Per-class total detections across all frames:")
    for k, v in counts.items():
        print(f"  {k:12s}  {v:6d}  (avg/frame: {v/n:.2f})")
    print(f"\nFrames with a ball detection: {ball_present}/{n} "
          f"({ball_present/n*100:.1f}%)")

    # --- Ball confidence distribution -------------------------------------
    ball_confs = []
    ball_in_player = 0
    ball_size_px = []
    for ft in tracks:
        ball = ft.get("ball", {}).get(1)
        if not ball:
            continue
        ball_confs.append(ball["confidence"])
        bx1, by1, bx2, by2 = ball["bbox"]
        ball_size_px.append(((bx2 - bx1) + (by2 - by1)) / 2)

        # Is this ball center inside any player bbox?
        bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
        for info in ft.get("player", {}).values():
            px1, py1, px2, py2 = info["bbox"]
            if px1 <= bcx <= px2 and py1 <= bcy <= py2:
                ball_in_player += 1
                break

    if ball_confs:
        ba = np.array(ball_confs)
        sa = np.array(ball_size_px)
        print("\nBall detection stats:")
        print(f"  confidence: min {ba.min():.2f}  median {np.median(ba):.2f}  "
              f"max {ba.max():.2f}  mean {ba.mean():.2f}")
        print(f"  size (px):  min {sa.min():.1f}  median {np.median(sa):.1f}  "
              f"max {sa.max():.1f}")
        print(f"  ball center INSIDE a player bbox: {ball_in_player}/{len(ball_confs)} "
              f"({ball_in_player/len(ball_confs)*100:.1f}%)")
        print("    ^ high % suggests white-shoe / kit-misclassification")

        bins = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01]
        hist, _ = np.histogram(ba, bins=bins)
        print("\n  conf histogram:")
        for lo, hi, c in zip(bins[:-1], bins[1:], hist):
            bar = "#" * int(40 * c / max(hist.max(), 1))
            print(f"    {lo:.2f}-{hi:.2f}  {c:4d}  {bar}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stubs/clip5.pkl")
