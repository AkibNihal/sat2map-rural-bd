"""
Split the collected pairs into train / val / test.

The split is written back into the manifest as a 'split' column and the
files are copied into train/, val/ and test/ subfolders, so the training
notebook can point at a folder and get the same tiles every run. Nothing
is shuffled at load time.

Two splits are produced:

  split            random 80/10/10, the primary result. Tiles are sampled
                   independently across the country with a dedup grid, so
                   there is no spatial adjacency between them and a random
                   split does not leak.

  geo_split        a secondary holdout where whole regions are reserved for
                   test. This measures generalisation to unseen parts of
                   the country rather than to unseen tiles from seen parts.
                   Recorded as a column only; no files are copied. Evaluate
                   the same trained models against it for a robustness line
                   in the results.

Run after collection has finished. Rerunning with the same SEED reproduces
the same assignment exactly.

With 3,000 pairs this gives the 2400 / 300 / 300 split used in the thesis.
Point the notebooks' DATA_ROOT at rural_dataset/<RUN_NAME>/.

Run:   python split_dataset.py
Needs: nothing outside the standard library
"""

import os
import csv
import math
import random
import shutil
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

RUN_NAME = "rural_z18_s512"     # folder under rural_dataset/
OUT_DIR = "rural_dataset"

SEED = 1907096                  # fixed, so the split is reproducible

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10                 # test gets the remainder

COPY_FILES = True               # False to only write the manifest column

# Regions reserved entirely for the geographic holdout. These are matched
# against the nearest_town column, so list towns rather than divisions.
# Roughly the Sylhet and Rangpur ends of the country.
GEO_HOLDOUT_TOWNS = {
    "Sylhet", "Maulavi Bazar", "Habiganj", "Sunamganj", "Baniachang",
    "Rangpur", "Saidpur", "Nageswari", "Chilmari", "Badarganj",
    "Thakurgaon", "Pirgaaj", "Dinajpur", "Panchagarh", "Kurigram",
    "Lalmonirhat", "Nilphamari", "Gaibandha",
}


try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

ROOT = os.path.join(BASE_DIR, OUT_DIR, RUN_NAME)
PAIR_DIR = os.path.join(ROOT, "pairs")
MANIFEST_PATH = os.path.join(ROOT, "manifest.csv")


# ---------------------------------------------------------------------------

def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit("Missing %s" % MANIFEST_PATH)
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("%s has no rows" % MANIFEST_PATH)
    return rows


def assign_random(rows):
    """Deterministic 80/10/10 over a seeded shuffle of the filenames."""
    names = sorted(r["filename"] for r in rows)     # sort first, so the
    rng = random.Random(SEED)                       # order does not depend
    rng.shuffle(names)                              # on directory listing

    n = len(names)
    n_train = int(round(TRAIN_FRAC * n))
    n_val = int(round(VAL_FRAC * n))

    split_of = {}
    for i, name in enumerate(names):
        if i < n_train:
            split_of[name] = "train"
        elif i < n_train + n_val:
            split_of[name] = "val"
        else:
            split_of[name] = "test"
    return split_of


def assign_geographic(rows):
    """Whole-region holdout, keyed on nearest_town."""
    geo_of = {}
    for r in rows:
        town = (r.get("nearest_town") or "").strip()
        geo_of[r["filename"]] = "test" if town in GEO_HOLDOUT_TOWNS else "train"
    return geo_of


def copy_into_folders(rows, split_of):
    for name in ("train", "val", "test"):
        d = os.path.join(ROOT, name)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    missing = 0
    for r in rows:
        fn = r["filename"]
        src = os.path.join(PAIR_DIR, fn)
        if not os.path.exists(src):
            missing += 1
            continue
        shutil.copy2(src, os.path.join(ROOT, split_of[fn], fn))
    if missing:
        print("Warning: %d manifest rows had no matching file" % missing)


def write_manifest(rows, split_of, geo_of):
    fields = list(rows[0].keys())
    for extra in ("split", "geo_split"):
        if extra not in fields:
            fields.append(extra)

    for r in rows:
        r["split"] = split_of[r["filename"]]
        r["geo_split"] = geo_of[r["filename"]]

    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    shutil.move(tmp, MANIFEST_PATH)


def summarise(rows, split_of, geo_of):
    counts = Counter(split_of.values())
    print("\nRandom split (primary):")
    for k in ("train", "val", "test"):
        print("  %-6s %5d" % (k, counts[k]))

    # road density per split, to check the split is not accidentally skewed
    by_split = defaultdict(list)
    for r in rows:
        try:
            by_split[split_of[r["filename"]]].append(float(r["map_detail"]))
        except (KeyError, ValueError):
            continue
    print("\n  map_detail by split:")
    print("  %-6s %8s %8s %8s" % ("split", "mean", "min", "max"))
    for k in ("train", "val", "test"):
        v = by_split[k]
        if v:
            print("  %-6s %8.5f %8.5f %8.5f"
                  % (k, sum(v) / len(v), min(v), max(v)))

    gcounts = Counter(geo_of.values())
    print("\nGeographic holdout (secondary, column only):")
    print("  train  %5d" % gcounts["train"])
    print("  test   %5d" % gcounts["test"])
    if gcounts["test"] == 0:
        print("  No tiles matched GEO_HOLDOUT_TOWNS. Check the town names "
              "against the nearest_town column.")
    elif gcounts["test"] < 30:
        print("  Only %d held out, which is thin for a robustness check. "
              "Add more towns to GEO_HOLDOUT_TOWNS." % gcounts["test"])

    towns = Counter((r.get("nearest_town") or "").strip() for r in rows)
    print("\nTop 10 nearest towns across the whole set:")
    for name, c in towns.most_common(10):
        print("  %-24s %4d" % (name, c))


def main():
    rows = load_manifest()
    print("Loaded %d rows from %s" % (len(rows), MANIFEST_PATH))

    split_of = assign_random(rows)
    geo_of = assign_geographic(rows)

    if COPY_FILES:
        copy_into_folders(rows, split_of)
        print("Copied files into train/, val/ and test/ under %s" % ROOT)

    write_manifest(rows, split_of, geo_of)
    print("Manifest updated with 'split' and 'geo_split' columns")

    summarise(rows, split_of, geo_of)

    print("\nSeed %d. Rerunning reproduces this split exactly." % SEED)


if __name__ == "__main__":
    main()