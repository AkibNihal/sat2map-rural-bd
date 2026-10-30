"""Final evaluation protocol (the one behind every reported number).

For each trained generator:

1. Predict a one-channel road score per tile, with test-time augmentation
   over four flips. RGB head: minimum of the three colour channels in [0, 1].
   Segmentation head: the road-class probability.
2. Threshold the score and apply a 3x3 morphological closing.
3. Choose the threshold on the VALIDATION split (best mean buffered IoU),
   freeze it, and score the TEST split once.

Metrics, all at 256x256:

    buffered IoU   both masks dilated by 2 px before IoU (main metric)
    strict IoU     no tolerance
    precision      predicted road within 2 px of true road
    recall         true road within 2 px of predicted road
    density ratio  total predicted road pixels / total true road pixels

Per-image values are averaged over tiles; pooled values sum intersections and
unions over the split first.
"""

import numpy as np
import cv2
import tensorflow as tf

from .models import ROAD

EVAL_SIZE = 256
BUFFER_PX = 2
TAR_THRESH = 0.90
CLOSE_PX = 1

THRESHOLDS = {
    "rgb": [round(0.80 + 0.01 * i, 2) for i in range(16)],   # 0.80 .. 0.95
    "seg": [round(0.15 + 0.05 * i, 2) for i in range(15)],   # 0.15 .. 0.85
}
FIXED_THRESH = {"rgb": 0.90, "seg": 0.50}

KEYS = ["b_int", "b_uni", "s_int", "s_uni", "p_hit", "p_tot",
        "r_hit", "r_tot", "n_pix"]


def _kernel(radius):
    return np.ones((2 * radius + 1,) * 2, np.uint8)


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------

def target_masks(ds, eval_size=EVAL_SIZE, tar_thresh=TAR_THRESH):
    """Boolean road masks of every target in a (batch 1) dataset."""
    out = []
    for _, tar in ds:
        if tar.shape[1] != eval_size:
            tar = tf.image.resize(tar, [eval_size, eval_size], method="area")
        t01 = (tar * 0.5 + 0.5).numpy()
        out.append(t01.min(axis=-1)[0] >= tar_thresh)
    return np.stack(out)


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------

def forward(gen, src, tta=True):
    """Generator output, averaged over identity, LR, UD and LR+UD flips."""
    out = gen(src, training=False)
    if tta:
        acc = out
        for flip in ("lr", "ud", "both"):
            s = src
            if flip in ("lr", "both"):
                s = tf.image.flip_left_right(s)
            if flip in ("ud", "both"):
                s = tf.image.flip_up_down(s)
            o = gen(s, training=False)
            if flip in ("lr", "both"):
                o = tf.image.flip_left_right(o)
            if flip in ("ud", "both"):
                o = tf.image.flip_up_down(o)
            acc = acc + o
        out = acc / 4.0
    return out


def road_score(gen, ds, head="rgb", eval_size=EVAL_SIZE, tta=True,
               keep_idx=()):
    """One-channel road score per tile, shape (N, eval_size, eval_size).

    TTA averages the raw output before the reduction, since the minimum of an
    average is not the average of the minimums. Tiles listed in keep_idx also
    have their full output returned (uint8 RGB for the rgb head, raw
    probabilities for seg), for figures.
    """
    scores, kept = [], {}
    for i, (src, _) in enumerate(ds):
        out = forward(gen, src, tta)
        if out.shape[1] != eval_size:
            out = tf.image.resize(out, [eval_size, eval_size], method="area")
        if head == "seg":
            o = out.numpy()
            scores.append(o[0, ..., ROAD])
            if i in keep_idx:
                kept[i] = o[0]
        else:
            o01 = (out * 0.5 + 0.5).numpy()
            scores.append(o01.min(axis=-1)[0])
            if i in keep_idx:
                kept[i] = np.clip(o01[0] * 255.0, 0, 255).astype(np.uint8)
    return np.stack(scores), kept


# ---------------------------------------------------------------------------
# masks and counts
# ---------------------------------------------------------------------------

def pred_mask(score_img, thresh, closing=True, close_px=CLOSE_PX):
    m = score_img >= thresh
    if closing:
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                             _kernel(close_px)) > 0
    return m


def counts(pred, true, buffer_px=BUFFER_PX):
    """Raw pixel counts for one tile; ratios are formed later."""
    bk = _kernel(buffer_px)
    pd_ = cv2.dilate(pred.astype(np.uint8), bk) > 0
    td_ = cv2.dilate(true.astype(np.uint8), bk) > 0
    return dict(
        b_int=int(np.logical_and(pd_, td_).sum()),
        b_uni=int(np.logical_or(pd_, td_).sum()),
        s_int=int(np.logical_and(pred, true).sum()),
        s_uni=int(np.logical_or(pred, true).sum()),
        p_hit=int(np.logical_and(pred, td_).sum()),
        p_tot=int(pred.sum()),
        r_hit=int(np.logical_and(pd_, true).sum()),
        r_tot=int(true.sum()),
        n_pix=int(pred.size),
    )


def ratio(a, b):
    return float(a) / float(b) if b else np.nan


def collect(score_arr, tar_arr, thresh, closing=True):
    """Per-tile metrics and summed counts at one threshold."""
    per = {"iou": [], "strict": [], "prec": [], "rec": [], "frac": []}
    tot = {k: 0 for k in KEYS}
    for i in range(score_arr.shape[0]):
        c = counts(pred_mask(score_arr[i], thresh, closing), tar_arr[i])
        for k in KEYS:
            tot[k] += c[k]
        per["iou"].append(ratio(c["b_int"], c["b_uni"]))
        per["strict"].append(ratio(c["s_int"], c["s_uni"]))
        per["prec"].append(ratio(c["p_hit"], c["p_tot"]))
        per["rec"].append(ratio(c["r_hit"], c["r_tot"]))
        per["frac"].append(c["p_tot"] / c["n_pix"])
    return per, tot


def summarise(per, tot):
    return dict(
        iou_perimg=float(np.nanmean(per["iou"])),
        iou_pooled=ratio(tot["b_int"], tot["b_uni"]),
        strict_perimg=float(np.nanmean(per["strict"])),
        strict_pooled=ratio(tot["s_int"], tot["s_uni"]),
        prec_perimg=float(np.nanmean(per["prec"])),
        prec_pooled=ratio(tot["p_hit"], tot["p_tot"]),
        rec_perimg=float(np.nanmean(per["rec"])),
        rec_pooled=ratio(tot["r_hit"], tot["r_tot"]),
        pred_frac=float(np.nanmean(per["frac"])),
        density_ratio=ratio(tot["p_tot"], tot["r_tot"]),
    )


# ---------------------------------------------------------------------------
# threshold selection and comparison
# ---------------------------------------------------------------------------

def sweep_threshold(val_score, val_tar, grid, closing=True):
    """Pick the grid value with the best mean buffered IoU on validation.

    Returns (best_thresh, best_iou, sweep) where sweep is a list of
    (thresh, summary) for every grid value. Ties go to the first value.
    """
    best_th, best_iou, sweep = None, -1.0, []
    for th in grid:
        s = summarise(*collect(val_score, val_tar, th, closing))
        sweep.append((th, s))
        if s["iou_perimg"] > best_iou:
            best_th, best_iou = th, s["iou_perimg"]
    return best_th, best_iou, sweep


def paired_bootstrap(a, b, n_boot=2000, rng=None):
    """Mean of (a - b) over tiles, with a 95% bootstrap interval.

    a, b are per-tile scores on the same tiles. NaN pairs are dropped. Pass
    one shared `rng` across calls to reproduce the reported intervals exactly
    (the notebooks create np.random.default_rng(0) once).
    Returns dict(mean_diff, ci_low, ci_high, frac_better, n_tiles, diffs).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    d = np.asarray(a, float) - np.asarray(b, float)
    dv = d[~np.isnan(d)]
    idx = rng.integers(0, dv.shape[0], size=(n_boot, dv.shape[0]))
    lo, hi = np.percentile(dv[idx].mean(axis=1), [2.5, 97.5])
    return dict(mean_diff=float(dv.mean()), ci_low=float(lo), ci_high=float(hi),
                frac_better=float((dv > 0).mean()), n_tiles=int(dv.shape[0]),
                diffs=dv)
