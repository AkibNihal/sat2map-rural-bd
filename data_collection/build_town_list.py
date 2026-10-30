"""
Build an urban exclusion list for Bangladesh from the GeoNames country file.

Downloads BD.zip from GeoNames once, keeps populated places above a
population threshold, and writes towns_bd.csv next to this script.
fetch_rural_pairs.py reads that CSV instead of a hand-typed city list.

The exclusion radius scales with population rather than being fixed, so
Dhaka gets a wide berth and a small upazila town gets a narrow one. A tile
covers about 1.3 km at zoom 16, so even the smallest radius clears the
built-up edge of a small town.

GeoNames data is CC BY 4.0. Cite it in the dataset section.

Run:   python build_town_list.py
Needs: requests
"""

import io
import os
import csv
import math
import zipfile

import requests


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

GEONAMES_URL = "https://download.geonames.org/export/dump/BD.zip"
OUT_FILE = "towns_bd.csv"
CACHE_FILE = "BD.txt"

MIN_POPULATION = 20000      # below this a place is treated as rural

# Radius scales as sqrt(population), anchored so that a place of
# REF_POPULATION gets REF_RADIUS_KM. Clamped to the min and max below.
REF_POPULATION = 100000
REF_RADIUS_KM = 6.0
MIN_RADIUS_KM = 3.0
MAX_RADIUS_KM = 25.0

# GeoNames feature codes worth excluding. P is populated places; PPLX is a
# section of a populated place, PPLL a locality that is often just a village.
KEEP_CODES = {"PPL", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLC", "PPLG"}

HEADERS = {"User-Agent": "sat2map-thesis/1.0 (KUET CSE; academic research)"}

# GeoNames dump column order, from their readme.
COL_NAME = 1
COL_ASCII = 2
COL_LAT = 4
COL_LON = 5
COL_FCLASS = 6
COL_FCODE = 7
COL_ADMIN1 = 10
COL_POP = 14

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

CACHE_PATH = os.path.join(BASE_DIR, CACHE_FILE)
OUT_PATH = os.path.join(BASE_DIR, OUT_FILE)


# ---------------------------------------------------------------------------

def download_geonames():
    """Fetch BD.zip and extract BD.txt next to this script."""
    print("Downloading %s ..." % GEONAMES_URL)
    r = requests.get(GEONAMES_URL, headers=HEADERS, timeout=180)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        with zf.open("BD.txt") as src, open(CACHE_PATH, "wb") as dst:
            dst.write(src.read())
    print("Saved %s (%.1f MB)"
          % (CACHE_PATH, os.path.getsize(CACHE_PATH) / 1e6))


def radius_for(population):
    """Exclusion radius in km, growing as the square root of population."""
    r = REF_RADIUS_KM * math.sqrt(population / float(REF_POPULATION))
    return max(MIN_RADIUS_KM, min(MAX_RADIUS_KM, r))


def build():
    if not os.path.exists(CACHE_PATH):
        download_geonames()
    else:
        print("Using cached %s" % CACHE_PATH)

    rows = []
    skipped_small = 0
    skipped_code = 0

    with open(CACHE_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue
            if parts[COL_FCLASS] != "P":
                continue
            if parts[COL_FCODE] not in KEEP_CODES:
                skipped_code += 1
                continue
            try:
                pop = int(parts[COL_POP] or 0)
                lat = float(parts[COL_LAT])
                lon = float(parts[COL_LON])
            except ValueError:
                continue
            if pop < MIN_POPULATION:
                skipped_small += 1
                continue
            rows.append({
                "name": parts[COL_ASCII] or parts[COL_NAME],
                "lat": "%.5f" % lat,
                "lon": "%.5f" % lon,
                "population": pop,
                "radius_km": "%.1f" % radius_for(pop),
                "admin1": parts[COL_ADMIN1],
                "fcode": parts[COL_FCODE],
            })

    rows.sort(key=lambda r: -r["population"])

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "lat", "lon", "population",
                                           "radius_km", "admin1", "fcode"])
        w.writeheader()
        w.writerows(rows)

    print("\nKept %d places at or above population %d" % (len(rows),
                                                          MIN_POPULATION))
    print("Dropped %d below the threshold, %d on feature code"
          % (skipped_small, skipped_code))
    print("Wrote %s" % OUT_PATH)

    print("\nLargest 15:")
    print("%-24s %10s %10s" % ("name", "population", "radius_km"))
    for r in rows[:15]:
        print("%-24s %10d %10s" % (r["name"], r["population"], r["radius_km"]))

    print("\nSmallest 10 kept:")
    print("%-24s %10s %10s" % ("name", "population", "radius_km"))
    for r in rows[-10:]:
        print("%-24s %10d %10s" % (r["name"], r["population"], r["radius_km"]))

    total_excluded = sum(math.pi * float(r["radius_km"]) ** 2 for r in rows)
    print("\nCircles cover about %.0f km2 before overlap, against 148460 km2 "
          "of Bangladesh." % total_excluded)
    print("Overlap between nearby towns means the true figure is lower.")


if __name__ == "__main__":
    build()
