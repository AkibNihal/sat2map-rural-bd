"""
Resolution check. Fetches five zoom-18 satellite tiles at full resolution and
writes each next to a copy downsampled to 256 px and scaled back up, so you
can judge how much road detail survives training at 256.

Writes fullres_check/z<ZOOM>/<name>_full.png and <name>_at256.png.

Run:   python check_fullres.py
Needs: requests, numpy, opencv-python, a Google Maps Static API key
"""

import os
import time

import cv2
import numpy as np
import requests

API_KEY = "PASTE_YOUR_KEY_HERE"  # https://console.cloud.google.com/apis/credentials

ZOOM = 18
SCALE = 2
REQ_SIZE = 640
CROP_PX = 55            # strips the attribution band, as in fetch_rural_pairs.py

# five z18 coordinates taken from the collection manifest
COORDS = [
    (26.02338, 88.56499, "thakurgaon"),
    (24.93699, 90.16501, "sherpur"),
    (24.02643, 89.41127, "pabna"),
    (24.92199, 90.58690, "netrakona"),
    (23.06251, 90.67702, "raipur"),
]

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
OUT = os.path.join(BASE_DIR, "fullres_check", "z%d" % ZOOM)


def fetch(lat, lon):
    params = {
        "center": "%s,%s" % (lat, lon), "zoom": ZOOM,
        "size": "%dx%d" % (REQ_SIZE, REQ_SIZE), "scale": SCALE,
        "maptype": "satellite", "format": "png32", "key": API_KEY,
    }
    r = requests.get("https://maps.googleapis.com/maps/api/staticmap",
                     params=params, timeout=30)
    if r.status_code != 200:
        print("  HTTP %d: %s" % (r.status_code, r.text[:150]))
        return None
    arr = np.frombuffer(r.content, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def crop(img):
    h = img.shape[0]
    img = img[CROP_PX:h - CROP_PX, :]
    h2, w2 = img.shape[:2]
    side = min(h2, w2)
    return img[(h2 - side) // 2:(h2 - side) // 2 + side,
               (w2 - side) // 2:(w2 - side) // 2 + side]


def main():
    os.makedirs(OUT, exist_ok=True)
    for lat, lon, name in COORDS:
        print(name)
        raw = fetch(lat, lon)
        time.sleep(0.2)
        if raw is None:
            continue
        full = crop(raw)
        small = cv2.resize(full, (256, 256), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(small, (full.shape[1], full.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUT, "%s_full.png" % name), full)
        cv2.imwrite(os.path.join(OUT, "%s_at256.png" % name), upscaled)
        print("  wrote %s_full.png and %s_at256.png (%dpx)"
              % (name, name, full.shape[0]))
    print("\nOpen the pairs in %s" % OUT)


if __name__ == "__main__":
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        raise SystemExit("Set API_KEY at the top of the file first.")
    main()
