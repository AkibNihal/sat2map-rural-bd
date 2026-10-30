"""Segmentation formulation.

The forced map palette has three exact colours, so a target tile is really a
three-class label map. Here the generator predicts those classes directly
(softmax head, see `models.make_generator(..., head="seg")`) and is trained
with soft Dice plus class-weighted cross-entropy. The map tile is recovered by
argmax and a palette lookup.

Labels are derived from the RGB targets on the fly, so no mask files are
needed, and the evaluation target is the same RGB colour test as the image
formulation, so both formulations are scored on the same quantity.
"""

import numpy as np
import tensorflow as tf

from .data import road_mask_tf
from .models import N_CLASS, ROAD, WATER

# the three forced palette colours, in [0, 1]
PALETTE01 = np.array([[208, 204, 196],      # background 0xd0ccc4
                      [255, 255, 255],      # road       0xffffff
                      [158, 203, 232]],     # water      0x9ecbe8
                     np.float32) / 255.0
PALETTE_TF = tf.constant(PALETTE01)

DICE_SMOOTH = 1.0


def labels_from_target(tar_pm1, road_thresh=0.98, water_delta=0.10):
    """(B, H, W) int32 labels from a [-1, 1] target batch.

    Road: every channel at least road_thresh (exact white).
    Water: blue minus red at least water_delta (0.29 for the water colour,
    0.00 for road, -0.05 for background), and not road.
    Everything else is background.
    """
    t01 = tar_pm1 * 0.5 + 0.5
    road = road_mask_tf(tar_pm1, road_thresh)
    water = (t01[..., 2] - t01[..., 0]) >= water_delta
    water = tf.logical_and(water, tf.logical_not(road))
    lab = tf.zeros(tf.shape(road), tf.int32)
    lab = tf.where(road, tf.fill(tf.shape(lab), ROAD), lab)
    lab = tf.where(water, tf.fill(tf.shape(lab), WATER), lab)
    return lab


def render_palette(probs):
    """Argmax over classes, then palette lookup. RGB in [0, 1]."""
    return tf.gather(PALETTE_TF, tf.argmax(probs, axis=-1))


def soft_render(probs):
    """Differentiable palette render, probabilities to RGB in [-1, 1].

    Used to feed the discriminator when an adversarial term is on.
    """
    rgb01 = tf.tensordot(probs, PALETTE_TF, axes=[[-1], [0]])
    return rgb01 * 2.0 - 1.0


def dice_loss(y_oh, y_pred, class_w):
    """Soft Dice per class, pooled over the batch, class-weighted mean of 1 - Dice."""
    inter = tf.reduce_sum(y_oh * y_pred, axis=[0, 1, 2])
    denom = tf.reduce_sum(y_oh + y_pred, axis=[0, 1, 2])
    dice = (2.0 * inter + DICE_SMOOTH) / (denom + DICE_SMOOTH)
    return tf.reduce_sum(class_w * (1.0 - dice)) / tf.reduce_sum(class_w)


def weighted_ce(y_oh, y_pred, class_w):
    """Cross-entropy with each pixel weighted by its true class weight."""
    pw = tf.reduce_sum(y_oh * class_w, axis=-1)
    ce = -tf.reduce_sum(y_oh * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)),
                        axis=-1)
    return tf.reduce_sum(pw * ce) / (tf.reduce_sum(pw) + 1e-7)


_bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def make_seg_train_step(generator, discriminator, g_opt, d_opt,
                        class_w=(0.2, 1.0, 0.5), dice_weight=1.0, ce_weight=1.0,
                        adv_weight=0.0, road_thresh=0.98, water_delta=0.10):
    """Compiled train step returning (total, dice, ce, adv, d_loss).

    With adv_weight 0 (the thesis runs) the discriminator is never used and
    may be None.
    """
    use_adv = adv_weight > 0
    cw = tf.constant(class_w, tf.float32)

    @tf.function
    def train_step(src, tar):
        labels = labels_from_target(tar, road_thresh, water_delta)
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            probs = generator(src, training=True)
            y_oh = tf.one_hot(labels, N_CLASS)
            d = dice_loss(y_oh, probs, cw)
            c = weighted_ce(y_oh, probs, cw)
            if use_adv:
                d_real = discriminator([src, tar], training=True)
                d_fake = discriminator([src, soft_render(probs)], training=True)
                d_loss = (_bce(tf.ones_like(d_real), d_real)
                          + _bce(tf.zeros_like(d_fake), d_fake))
                adv = _bce(tf.ones_like(d_fake), d_fake)
            else:
                d_loss, adv = tf.constant(0.0), tf.constant(0.0)
            total = dice_weight * d + ce_weight * c + adv_weight * adv

        g_opt.apply_gradients(zip(
            gt.gradient(total, generator.trainable_variables),
            generator.trainable_variables))
        if use_adv:
            d_opt.apply_gradients(zip(
                dt.gradient(d_loss, discriminator.trainable_variables),
                discriminator.trainable_variables))
        return total, d, c, adv, d_loss

    return train_step


def quick_eval_seg(generator, ds, eval_size=256, prob_thresh=0.50,
                   tar_thresh=0.90, buffer_px=2):
    """Mean strict and buffered IoU with the road probability at a fixed
    threshold. Used for checkpoint selection; final numbers come from
    `evaluation.py`."""
    from .training import iou_pair
    strict, buffered = [], []
    for src, tar in ds:
        probs = generator(src, training=False)
        if probs.shape[1] != eval_size:
            probs = tf.image.resize(probs, [eval_size, eval_size], method="area")
            tar = tf.image.resize(tar, [eval_size, eval_size], method="area")
        pm = probs[..., ROAD].numpy() >= prob_thresh
        tm = (tar * 0.5 + 0.5).numpy().min(axis=-1) >= tar_thresh
        for j in range(pm.shape[0]):
            strict.append(iou_pair(pm[j], tm[j], 0))
            buffered.append(iou_pair(pm[j], tm[j], buffer_px))
    return float(np.nanmean(strict)), float(np.nanmean(buffered))
