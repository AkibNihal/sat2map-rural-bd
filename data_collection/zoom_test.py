"""
Zoom sweep. Fetches the same rural coordinates at several zoom levels from
the Google Maps Static API, to pick a scale before collecting the dataset
(the thesis settled on zoom 18).

For each zoom it writes:
    zoom_test/z<zoom>/<name>_pair.png    satellite | map, at OUT_SIZE
    zoom_test/z<zoom>/<name>_map.png     the map half on its own
    zoom_test/compare_<name>.png         one coordinate across all zooms
    zoom_test/zoom_stats.csv             per-zoom detail numbers

Cost is len(COORDS) * len(ZOOMS) * 2 API calls. With the defaults that is
40 calls, well inside the monthly free tier.

Run:   python zoom_test.py
Needs: requests, numpy, opencv-python, matplotlib, a Maps Static API key
"""

import os
import csv
import math
import time

import cv2
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

API_KEY = "PASTE_YOUR_KEY_HERE"  # get one from https://console.cloud.google.com/apis/credentials

ZOOMS = [15, 16, 17, 18]

SCALE = 2
REQ_SIZE = 640
CROP_PX = 55
OUT_SIZE = 256

OUT_DIR = "zoom_test"
SLEEP_BETWEEN = 0.15
MAX_RETRIES = 4
TIMEOUT = 30

# Rural coordinates. Mix ones you know have road content with a couple of
# sparse ones, so you can see how each zoom handles both.
COORDS = [
    (23.40369, 91.40429, "cumilla_01"),
    (23.43334, 91.38040, "cumilla_02"),
    (22.72000, 89.08000, "satkhira"),
    (24.60000, 90.30000, "mymensingh"),
    (25.60000, 89.15000, "rangpur"),
]

MAP_STYLES = [
    "feature:all|element:labels|visibility:off",
    "feature:poi|visibility:off",
    "feature:transit|visibility:off",
    "feature:administrative|visibility:off",
    "feature:landscape|element:geometry|color:0xd0ccc4",
    "feature:water|element:geometry|color:0x9ecbe8",
    "feature:road|element:geometry.fill|color:0xffffff",
    "feature:road|element:geometry.stroke|visibility:off",
]

BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

ROOT = os.path.join(BASE_DIR, OUT_DIR)


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------

def fetch(lat, lon, zoom, maptype):
    params = [
        ("center", "%s,%s" % (lat, lon)),
        ("zoom", str(zoom)),
        ("size", "%dx%d" % (REQ_SIZE, REQ_SIZE)),
        ("scale", str(SCALE)),
        ("maptype", maptype),
        ("format", "png32"),
        ("key", API_KEY),
    ]
    if maptype == "roadmap":
        params += [("style", s) for s in MAP_STYLES]

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
        except requests.RequestException as exc:
            print("    network error (%s)" % exc.__class__.__name__)
            time.sleep(2 ** attempt)
            continue

        if r.status_code == 200:
            if not r.headers.get("Content-Type", "").startswith("image"):
                print("    non-image: %s" % r.text[:180])
                return None
            if len(r.content) < 5000:
                print("    tiny image, treating as error")
                return None
            arr = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        print("    HTTP %d: %s" % (r.status_code, r.text[:180]))
        return None
    return None


# ---------------------------------------------------------------------------
# PROCESSING
# ---------------------------------------------------------------------------

def crop_and_resize(img):
    h = img.shape[0]
    img = img[CROP_PX:h - CROP_PX, :]
    h2, w2 = img.shape[:2]
    side = min(h2, w2)
    y0 = (h2 - side) // 2
    x0 = (w2 - side) // 2
    return cv2.resize(img[y0:y0 + side, x0:x0 + side],
                      (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)


def detail_fraction(img):
    """Fraction of pixels that are not the single most common colour."""
    q = (img // 16).astype(np.uint8).reshape(-1, 3)
    _, counts = np.unique(q, axis=0, return_counts=True)
    return 1.0 - counts.max() / float(q.shape[0])


def distinct_colours(img):
    """How many distinct quantised colours the map uses.

    Google drops whole road classes at wide zooms rather than just drawing
    them smaller, and this number tracks that."""
    q = (img // 16).astype(np.uint8).reshape(-1, 3)
    return int(np.unique(q, axis=0).shape[0])


def ground_metrics(zoom, lat):
    """Metres across the final tile, and metres per output pixel."""
    mpp_canvas = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom) / SCALE
    canvas_side = REQ_SIZE * SCALE - 2 * CROP_PX
    span_m = canvas_side * mpp_canvas
    return span_m, span_m / OUT_SIZE


def save_rgb(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

FIELDS = ["name", "lat", "lon", "zoom", "span_m", "m_per_px",
          "map_detail", "map_colours", "sat_detail"]


def main():
    os.makedirs(ROOT, exist_ok=True)
    rows = []
    store = {}          # (name, zoom) -> (sat, map)

    total = len(COORDS) * len(ZOOMS)
    done = 0

    for lat, lon, name in COORDS:
        for zoom in ZOOMS:
            done += 1
            print("[%d/%d] %s at zoom %d" % (done, total, name, zoom))

            map_raw = fetch(lat, lon, zoom, "roadmap")
            time.sleep(SLEEP_BETWEEN)
            if map_raw is None:
                print("    map fetch failed, skipping")
                continue

            sat_raw = fetch(lat, lon, zoom, "satellite")
            time.sleep(SLEEP_BETWEEN)
            if sat_raw is None:
                print("    satellite fetch failed, skipping")
                continue

            mp = crop_and_resize(map_raw)
            sat = crop_and_resize(sat_raw)
            store[(name, zoom)] = (sat, mp)

            zdir = os.path.join(ROOT, "z%d" % zoom)
            save_rgb(np.concatenate([sat, mp], axis=1),
                     os.path.join(zdir, "%s_pair.png" % name))
            save_rgb(mp, os.path.join(zdir, "%s_map.png" % name))

            span, mpp = ground_metrics(zoom, lat)
            rows.append({
                "name": name, "lat": lat, "lon": lon, "zoom": zoom,
                "span_m": "%.0f" % span, "m_per_px": "%.2f" % mpp,
                "map_detail": "%.4f" % detail_fraction(mp),
                "map_colours": distinct_colours(mp),
                "sat_detail": "%.4f" % detail_fraction(sat),
            })

    if not rows:
        print("Nothing collected. Check the API key and that Maps Static "
              "is enabled on the project.")
        return

    with open(os.path.join(ROOT, "zoom_stats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # one figure per coordinate, satellite on top, map below, zoom across
    for lat, lon, name in COORDS:
        zs = [z for z in ZOOMS if (name, z) in store]
        if not zs:
            continue
        fig, axes = plt.subplots(2, len(zs), figsize=(3.1 * len(zs), 6.6))
        axes = np.atleast_2d(axes)
        for c, z in enumerate(zs):
            sat, mp = store[(name, z)]
            span, mpp = ground_metrics(z, lat)
            axes[0, c].imshow(sat)
            axes[0, c].set_title("z%d\n%.0f m across, %.1f m/px"
                                 % (z, span, mpp), fontsize=9)
            axes[1, c].imshow(mp)
            for r in (0, 1):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
        axes[0, 0].set_ylabel("satellite", fontsize=9)
        axes[1, 0].set_ylabel("map target", fontsize=9)
        fig.suptitle("%s  (%.5f, %.5f)" % (name, lat, lon), fontsize=11)
        plt.tight_layout()
        out = os.path.join(ROOT, "compare_%s.png" % name)
        plt.savefig(out, dpi=110)
        plt.close(fig)
        print("wrote %s" % out)

    print()
    print("Per-zoom means across %d coordinates:" % len(COORDS))
    print("%6s %10s %10s %12s %12s" %
          ("zoom", "span_m", "m/px", "map_detail", "map_colours"))
    for z in ZOOMS:
        sel = [r for r in rows if r["zoom"] == z]
        if not sel:
            continue
        print("%6d %10.0f %10.2f %12.4f %12.1f" % (
            z,
            np.mean([float(r["span_m"]) for r in sel]),
            np.mean([float(r["m_per_px"]) for r in sel]),
            np.mean([float(r["map_detail"]) for r in sel]),
            np.mean([float(r["map_colours"]) for r in sel]),
        ))
    print()
    print("Output in %s" % ROOT)
    print("Open the compare_*.png files. map_detail falling as zoom widens "
          "is expected; map_colours falling means Google is dropping whole "
          "road classes at that scale, which is the worse problem.")


if __name__ == "__main__":
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        raise SystemExit("Set API_KEY at the top of the file first.")
    main()