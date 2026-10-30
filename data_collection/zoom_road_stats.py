"""
Road statistics for the zoom sweep. Reads the map tiles written by
zoom_test.py and prints, per zoom level, the old "detail" measure (share of
pixels that are not the dominant colour) next to the exact road fraction
(pixels whose darkest channel is at least 250, i.e. the forced white). Then
lists the six most common colours of a few z16 tiles, to confirm the palette.

Run zoom_test.py first.

Run:   python zoom_road_stats.py
Needs: numpy, opencv-python
"""

import glob
import os

import cv2
import numpy as np

try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()

ZOOMS = (15, 16, 17, 18)
PALETTE_TILES = ("cumilla_01", "cumilla_02", "mymensingh", "rangpur", "satkhira")


def road_frac(img):
    return float((img.astype(np.int16).min(axis=2) >= 250).mean())


def old_detail(img):
    q = (img // 16).astype(np.uint8).reshape(-1, 3)
    _, counts = np.unique(q, axis=0, return_counts=True)
    return 1.0 - counts.max() / float(q.shape[0])


def main():
    for z in ZOOMS:
        paths = sorted(glob.glob(os.path.join(BASE, "zoom_test", "z%d" % z, "*_map.png")))
        if not paths:
            continue
        print("\n--- zoom %d ---" % z)
        print("%-20s %10s %10s" % ("tile", "old", "road"))
        for p in paths:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            name = os.path.basename(p).replace("_map.png", "")
            print("%-20s %10.5f %10.5f" % (name, old_detail(img), road_frac(img)))

    for tile in PALETTE_TILES:
        p = os.path.join(BASE, "zoom_test", "z16", "%s_map.png" % tile)
        if not os.path.exists(p):
            continue
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        a = (img // 8 * 8).reshape(-1, 3)
        cols, counts = np.unique(a, axis=0, return_counts=True)
        print("\n--- %s z16 ---" % tile)
        for i in np.argsort(-counts)[:6]:
            print(cols[i], counts[i], "%.4f" % (counts[i] / len(a)))


if __name__ == "__main__":
    main()
