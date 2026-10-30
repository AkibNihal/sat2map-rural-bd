"""
Paired satellite / map tile collector for rural Bangladesh, using the
Google Maps Static API for both halves.

Produces concatenated PNGs, satellite on the left, styled roadmap on the
right, each half OUT_SIZE square. At OUT_SIZE = 512 the files are 1024x512.
The training pipeline (sat2map/data.py) splits each file at the midpoint of
its width and resizes both halves, so any stored size works.

Collecting at 512 costs no extra API calls and can be downsampled to 256 at
load time. Collecting at 256 and later wanting 512 means refetching, so
there is no reason to collect small.

Sampling is rural by construction:
  - points fall only inside the Bangladesh land boundary, eroded inward so
    no tile straddles the border
  - points inside any town's exclusion radius are rejected, using the
    GeoNames-derived list from build_town_list.py
  - tiles whose map target has too little road are rejected

The map styling forces a fixed palette, so the target does not change
colour with land cover and road pixels are exactly white. Road detection
is then an exact colour test rather than a heuristic.

The map is fetched before the satellite, so a rejected tile costs one API
call rather than two.

Output goes to ./rural_dataset/<mode>_z<zoom>_s<size>/ so runs at different
zoom levels or output sizes never collide.

Run build_town_list.py first, then this with PREVIEW_ONLY = True to check
one pair, then with PREVIEW_ONLY = False to collect. Collection resumes from
manifest.csv if interrupted.

Run:   python fetch_rural_pairs.py   
Needs: requests, numpy, opencv-python, and a Google Maps Static API key.
       Never commit the key: leave API_KEY as the placeholder in the repo.
"""

import os
import csv
import json
import math
import time
import random

import cv2
import numpy as np
import requests


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

API_KEY = "PASTE_YOUR_KEY_HERE"  # get one from https://console.cloud.google.com/apis/credentials

PREVIEW_ONLY = False
COLLECT_MODE = "rural"          # "rural" or "urban_control"

N_PAIRS = 3000                   # start at 100, raise once the rate is known
ZOOM = 18
SCALE = 2
REQ_SIZE = 640

CROP_PX = 55
OUT_SIZE = 512                  # per half; files are 2*OUT_SIZE wide

OUT_DIR = "rural_dataset"
BOUNDARY_FILE = "bangladesh_adm0.geojson"
TOWNS_FILE = "towns_bd.csv"

# Fraction of target pixels that must render as road. The z18 sample of 20
# ran 0.0043 to 0.0745 with a median near 0.029, so 0.004 keeps almost all
# of them while still excluding genuinely empty tiles.
MIN_MAP_DETAIL = 0.004
MIN_SAT_STD = 12.0              # rejects flat satellite tiles: water, cloud

BORDER_BUFFER_KM = 3.0

# Roughly one tile width, so samples do not overlap. At zoom 18 a tile spans
# about 320 m, which is 0.003 degrees.
DEDUP_DEG = 0.003
MASK_SIZE = 2400

SLEEP_BETWEEN = 0.12
MAX_RETRIES = 4
TIMEOUT = 30
MAX_ATTEMPT_MULTIPLIER = 20     # z18 rejects more, so allow more tries

BD_BBOX = (20.50, 88.00, 26.70, 92.70)   # south, west, north, east

BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"

# Forced palette. Labels, POI and transit are stripped because they cannot
# be inferred from satellite imagery. Road stroke is off so roads render as
# flat white with no darker outline.
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

HEADERS = {"User-Agent": "sat2map-thesis/1.0 (KUET CSE; academic research)"}
GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/BGD/ADM0/"


try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

RUN_NAME = "%s_z%d_s%d" % (
    "rural" if COLLECT_MODE == "rural" else "urban_control", ZOOM, OUT_SIZE)
ROOT = os.path.join(BASE_DIR, OUT_DIR, RUN_NAME)
PAIR_DIR = os.path.join(ROOT, "pairs")
MANIFEST_PATH = os.path.join(ROOT, "manifest.csv")
BOUNDARY_PATH = os.path.join(BASE_DIR, BOUNDARY_FILE)
TOWNS_PATH = os.path.join(BASE_DIR, TOWNS_FILE)

TOWNS = []
_TOWN_ARR = None


# ---------------------------------------------------------------------------
# TOWN EXCLUSION LIST
# ---------------------------------------------------------------------------

def load_towns():
    """Read the exclusion list written by build_town_list.py."""
    global TOWNS, _TOWN_ARR
    if not os.path.exists(TOWNS_PATH):
        raise SystemExit("Missing %s. Run build_town_list.py first."
                         % TOWNS_PATH)
    rows = []
    with open(TOWNS_PATH, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((row["name"], float(row["lat"]), float(row["lon"]),
                         float(row["radius_km"])))
    if not rows:
        raise SystemExit("%s is empty." % TOWNS_PATH)

    TOWNS = rows
    _TOWN_ARR = (
        np.radians(np.array([r[1] for r in rows])),
        np.radians(np.array([r[2] for r in rows])),
        np.array([r[3] for r in rows]),
    )
    print("Loaded %d towns for exclusion (radii %.1f to %.1f km)"
          % (len(rows), _TOWN_ARR[2].min(), _TOWN_ARR[2].max()))


def _distances_km(lat, lon):
    """Haversine distance from one point to every town, vectorised."""
    tlat, tlon, _ = _TOWN_ARR
    p = math.radians(lat)
    l = math.radians(lon)
    dp = tlat - p
    dl = tlon - l
    a = np.sin(dp / 2) ** 2 + math.cos(p) * np.cos(tlat) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def town_context(lat, lon):
    """(name, distance_km, that town's radius_km, inside_any)."""
    d = _distances_km(lat, lon)
    trad = _TOWN_ARR[2]
    i = int(np.argmin(d))
    return TOWNS[i][0], float(d[i]), float(trad[i]), bool(np.any(d < trad))


# ---------------------------------------------------------------------------
# BOUNDARY MASK
# ---------------------------------------------------------------------------

def download_boundary():
    print("Boundary file not found, downloading from geoBoundaries...")
    meta = requests.get(GEOBOUNDARIES_API, headers=HEADERS, timeout=60)
    meta.raise_for_status()
    url = meta.json()["gjDownloadURL"]
    gj = requests.get(url, headers=HEADERS, timeout=180)
    gj.raise_for_status()
    with open(BOUNDARY_PATH, "wb") as fh:
        fh.write(gj.content)
    print("Saved %s" % BOUNDARY_PATH)


def polygons_from_geojson(obj):
    def walk(geom):
        t = geom.get("type")
        if t == "Polygon":
            yield geom["coordinates"]
        elif t == "MultiPolygon":
            for poly in geom["coordinates"]:
                yield poly
        elif t == "GeometryCollection":
            for g in geom.get("geometries", []):
                for p in walk(g):
                    yield p

    if obj.get("type") == "FeatureCollection":
        for feat in obj["features"]:
            for p in walk(feat["geometry"]):
                yield p
    elif obj.get("type") == "Feature":
        for p in walk(obj["geometry"]):
            yield p
    else:
        for p in walk(obj):
            yield p


def build_land_mask():
    """Rasterise the national boundary into a boolean grid over BD_BBOX."""
    if not os.path.exists(BOUNDARY_PATH):
        try:
            download_boundary()
        except Exception as exc:
            raise SystemExit(
                "Could not download the boundary (%s).\n"
                "Get the Bangladesh ADM0 GeoJSON manually from "
                "geoboundaries.org and save it as %s" % (exc, BOUNDARY_PATH))

    with open(BOUNDARY_PATH, "r", encoding="utf-8") as fh:
        obj = json.load(fh)

    south, west, north, east = BD_BBOX
    mask = np.zeros((MASK_SIZE, MASK_SIZE), np.uint8)

    def to_px(lon, lat):
        x = (lon - west) / (east - west) * (MASK_SIZE - 1)
        y = (north - lat) / (north - south) * (MASK_SIZE - 1)
        return [int(round(x)), int(round(y))]

    n_poly = 0
    for rings in polygons_from_geojson(obj):
        outer = np.array([to_px(c[0], c[1]) for c in rings[0]], np.int32)
        cv2.fillPoly(mask, [outer], 1)
        for hole in rings[1:]:
            h = np.array([to_px(c[0], c[1]) for c in hole], np.int32)
            cv2.fillPoly(mask, [h], 0)
        n_poly += 1

    deg_per_px = (north - south) / float(MASK_SIZE)
    erode_px = max(1, int(round(BORDER_BUFFER_KM / (deg_per_px * 111.32))))
    kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    mask = cv2.erode(mask, kernel)

    print("Land mask: %d polygons, eroded %d px (~%.1f km), %d of %d cells land"
          % (n_poly, erode_px, BORDER_BUFFER_KM, int(mask.sum()), mask.size))
    return mask.astype(bool)


def on_land(mask, lat, lon):
    south, west, north, east = BD_BBOX
    if not (south <= lat <= north and west <= lon <= east):
        return False
    x = int(round((lon - west) / (east - west) * (MASK_SIZE - 1)))
    y = int(round((north - lat) / (north - south) * (MASK_SIZE - 1)))
    return bool(mask[y, x])


# ---------------------------------------------------------------------------
# SAMPLING
# ---------------------------------------------------------------------------

def sample_point(mask):
    """Rejection-sample a point satisfying the land and town rules."""
    south, west, north, east = BD_BBOX
    for _ in range(500):
        lat = round(random.uniform(south, north), 5)
        lon = round(random.uniform(west, east), 5)
        if not on_land(mask, lat, lon):
            continue
        name, km, rad, inside = town_context(lat, lon)
        if COLLECT_MODE == "rural":
            if inside:
                continue
        else:
            if not inside:
                continue
        return lat, lon, name, km, rad
    return None


def grid_key(lat, lon):
    return (int(lat / DEDUP_DEG), int(lon / DEDUP_DEG))


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------

def fetch(lat, lon, maptype):
    params = [
        ("center", "%s,%s" % (lat, lon)),
        ("zoom", str(ZOOM)),
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
            print("  network error (%s)" % exc.__class__.__name__)
            time.sleep(2 ** attempt)
            continue

        if r.status_code == 200:
            if not r.headers.get("Content-Type", "").startswith("image"):
                print("  non-image: %s" % r.text[:180])
                return None
            if len(r.content) < 5000:
                print("  tiny image, treating as error")
                return None
            arr = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        print("  HTTP %d: %s" % (r.status_code, r.text[:180]))
        return None
    return None


# ---------------------------------------------------------------------------
# PROCESSING
# ---------------------------------------------------------------------------

def crop_square(img):
    """Strip the attribution band and centre-crop to square, full resolution."""
    h = img.shape[0]
    img = img[CROP_PX:h - CROP_PX, :]
    h2, w2 = img.shape[:2]
    side = min(h2, w2)
    y0 = (h2 - side) // 2
    x0 = (w2 - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


def crop_and_resize(img):
    sq = crop_square(img)
    if sq.shape[0] == OUT_SIZE:
        return sq
    return cv2.resize(sq, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)


def map_detail_fraction(img):
    """Fraction of pixels rendering as road, given the forced white palette."""
    return float((img.astype(np.int16).min(axis=2) >= 250).mean())


def sat_std(img):
    return float(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).std())


def save_rgb(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# PREVIEW
# ---------------------------------------------------------------------------

def run_preview(mask):
    os.makedirs(ROOT, exist_ok=True)
    print("\nRun name: %s" % RUN_NAME)
    print("\nEight sample points, to check the rules:")
    for _ in range(8):
        got = sample_point(mask)
        if got is None:
            print("  no point found in 500 tries, loosen the rules")
            continue
        lat, lon, town, km, rad = got
        print("  %.5f, %.5f   nearest %s at %.1f km (its radius %.1f km)"
              % (lat, lon, town, km, rad))

    got = sample_point(mask)
    if got is None:
        return
    lat, lon, town, km, rad = got
    print("\nFetching a full pair at %.5f, %.5f" % (lat, lon))

    map_raw = fetch(lat, lon, "roadmap")
    time.sleep(SLEEP_BETWEEN)
    if map_raw is None:
        print("Map fetch failed. Check the key and that Maps Static is enabled.")
        return
    sat_raw = fetch(lat, lon, "satellite")
    if sat_raw is None:
        print("Satellite fetch failed.")
        return

    save_rgb(map_raw, os.path.join(ROOT, "preview_raw_map.png"))
    mp = crop_and_resize(map_raw)
    sat = crop_and_resize(sat_raw)
    save_rgb(mp, os.path.join(ROOT, "preview_map.png"))
    pair = np.concatenate([sat, mp], axis=1)
    save_rgb(pair, os.path.join(ROOT, "preview_pair.png"))

    print("pair size %dx%d" % (pair.shape[1], pair.shape[0]))
    print("map_detail %.5f (gate %.5f)   sat_std %.2f (gate %.1f)"
          % (map_detail_fraction(mp), MIN_MAP_DETAIL,
             sat_std(sat), MIN_SAT_STD))
    print("\nCheck preview_raw_map.png for where the attribution sits, then")
    print("preview_pair.png to confirm it is cropped out and the halves align.")
    print("When it looks right, set PREVIEW_ONLY = False.")


# ---------------------------------------------------------------------------
# COLLECTION
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "filename", "lat", "lon", "mode", "nearest_town", "town_dist_km",
    "town_radius_km", "zoom", "scale", "req_size", "crop_px", "out_size",
    "map_detail", "sat_std", "fetched_utc",
]


def load_existing():
    seen, n = set(), 0
    if not os.path.exists(MANIFEST_PATH):
        return seen, n
    with open(MANIFEST_PATH, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                seen.add(grid_key(float(row["lat"]), float(row["lon"])))
                n += 1
            except (KeyError, ValueError):
                continue
    return seen, n


def run_collection(mask):
    os.makedirs(PAIR_DIR, exist_ok=True)
    print("Run name: %s" % RUN_NAME)
    seen, saved = load_existing()
    if saved:
        print("Resuming from %d existing pairs." % saved)

    new_file = not os.path.exists(MANIFEST_PATH)
    fh = open(MANIFEST_PATH, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
    if new_file:
        writer.writeheader()

    attempts = 0
    calls = 0
    rejected = {"no_point": 0, "dup": 0, "fetch": 0,
                "blank_map": 0, "flat_sat": 0}
    max_attempts = N_PAIRS * MAX_ATTEMPT_MULTIPLIER

    try:
        while saved < N_PAIRS and attempts < max_attempts:
            attempts += 1

            got = sample_point(mask)
            if got is None:
                rejected["no_point"] += 1
                continue
            lat, lon, town, km, rad = got

            key = grid_key(lat, lon)
            if key in seen:
                rejected["dup"] += 1
                continue
            seen.add(key)

            # map first, so a blank target costs one call not two
            map_raw = fetch(lat, lon, "roadmap")
            calls += 1
            time.sleep(SLEEP_BETWEEN)
            if map_raw is None:
                rejected["fetch"] += 1
                continue

            mp = crop_and_resize(map_raw)
            detail = map_detail_fraction(mp)
            if detail < MIN_MAP_DETAIL:
                rejected["blank_map"] += 1
                continue

            sat_raw = fetch(lat, lon, "satellite")
            calls += 1
            time.sleep(SLEEP_BETWEEN)
            if sat_raw is None:
                rejected["fetch"] += 1
                continue

            sat = crop_and_resize(sat_raw)
            sstd = sat_std(sat)
            if sstd < MIN_SAT_STD:
                rejected["flat_sat"] += 1
                continue

            fname = "%s,%s_concat.png" % (lat, lon)
            save_rgb(np.concatenate([sat, mp], axis=1),
                     os.path.join(PAIR_DIR, fname))

            writer.writerow({
                "filename": fname, "lat": lat, "lon": lon,
                "mode": COLLECT_MODE, "nearest_town": town,
                "town_dist_km": "%.1f" % km, "town_radius_km": "%.1f" % rad,
                "zoom": ZOOM, "scale": SCALE, "req_size": REQ_SIZE,
                "crop_px": CROP_PX, "out_size": OUT_SIZE,
                "map_detail": "%.5f" % detail, "sat_std": "%.2f" % sstd,
                "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
            })
            fh.flush()
            saved += 1

            if saved % 25 == 0:
                print("%d/%d saved   %d attempts   %d API calls"
                      % (saved, N_PAIRS, attempts, calls))
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun to resume from the manifest.")
    finally:
        fh.close()

    print("\nSaved %d pairs to %s" % (saved, PAIR_DIR))
    print("API calls this run: %d" % calls)
    print("Rejected: %s" % rejected)
    if attempts:
        print("Acceptance rate: %.1f%%" % (100.0 * saved / attempts))
        if saved:
            print("Estimated cost per 1000 pairs: %.0f API calls"
                  % (1000.0 * calls / saved))


if __name__ == "__main__":
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        raise SystemExit("Set API_KEY at the top of the file first.")
    load_towns()
    land = build_land_mask()
    if PREVIEW_ONLY:
        run_preview(land)
    else:
        run_collection(land)