"""Losses, train step and the quick IoU used for checkpoint selection."""

import numpy as np
import cv2
import tensorflow as tf

from .data import road_mask_tf

_bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def weighted_mae(y_true, y_pred, road_weight, loss_thresh=0.98):
    """MAE with road pixels weighted by (1 + road_weight).

    The road mask comes from the target itself. Dividing by the mean weight
    keeps the loss scale comparable across road_weight values, so 0 (plain
    MAE) and 10 can be compared directly.
    """
    road = tf.cast(road_mask_tf(y_true, loss_thresh), tf.float32)
    w = tf.expand_dims(1.0 + road_weight * road, -1)
    return tf.reduce_mean(w * tf.abs(y_true - y_pred)) / tf.reduce_mean(w)


def discriminator_loss(d_real, d_fake):
    return (_bce(tf.ones_like(d_real), d_real)
            + _bce(tf.zeros_like(d_fake), d_fake))


def make_train_step(generator, discriminator, g_opt, d_opt, adv_weight,
                    l1_weight=100.0, road_weight=10.0, loss_thresh=0.98):
    """Return a compiled Pix2Pix train step.

    adv_weight 0 trains the generator with weighted L1 only and never updates
    the discriminator.
    """
    use_adv = adv_weight > 0

    @tf.function
    def train_step(src, tar):
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            gen = generator(src, training=True)
            if use_adv:
                d_real = discriminator([src, tar], training=True)
                d_fake = discriminator([src, gen], training=True)
                d_loss = discriminator_loss(d_real, d_fake)
                g_adv = _bce(tf.ones_like(d_fake), d_fake)
            else:
                d_loss = tf.constant(0.0)
                g_adv = tf.constant(0.0)
            g_l1 = weighted_mae(tar, gen, road_weight, loss_thresh)
            g_total = adv_weight * g_adv + l1_weight * g_l1

        g_opt.apply_gradients(zip(
            gt.gradient(g_total, generator.trainable_variables),
            generator.trainable_variables))
        if use_adv:
            d_opt.apply_gradients(zip(
                dt.gradient(d_loss, discriminator.trainable_variables),
                discriminator.trainable_variables))
        return g_total, g_adv, g_l1, d_loss

    return train_step


# ---------------------------------------------------------------------------
# quick IoU during training (fixed threshold, no TTA)
# ---------------------------------------------------------------------------

def iou_pair(pred, true, buffer_px=0):
    """IoU of two boolean masks, optionally after dilating both by buffer_px."""
    if buffer_px:
        k = np.ones((2 * buffer_px + 1,) * 2, np.uint8)
        pred = cv2.dilate(pred.astype(np.uint8), k) > 0
        true = cv2.dilate(true.astype(np.uint8), k) > 0
    union = np.logical_or(pred, true).sum()
    return float(np.logical_and(pred, true).sum()) / union if union else np.nan


def quick_eval(generator, ds, eval_size=256, gen_thresh=0.90, tar_thresh=0.90,
               buffer_px=2):
    """Mean strict and buffered IoU over a dataset, at a fixed threshold.

    Used every few epochs to pick the best checkpoint. Outputs and targets are
    resized to eval_size first, so a 512 model is scored on the same grid as a
    256 one. The final numbers come from `evaluation.py`, not from here.
    """
    strict, buffered = [], []
    for src, tar in ds:
        gen = generator(src, training=False)
        if gen.shape[1] != eval_size:
            gen = tf.image.resize(gen, [eval_size, eval_size], method="area")
            tar = tf.image.resize(tar, [eval_size, eval_size], method="area")
        gm = (gen * 0.5 + 0.5).numpy().min(axis=-1) >= gen_thresh
        tm = (tar * 0.5 + 0.5).numpy().min(axis=-1) >= tar_thresh
        for j in range(gm.shape[0]):
            strict.append(iou_pair(gm[j], tm[j], 0))
            buffered.append(iou_pair(gm[j], tm[j], buffer_px))
    return float(np.nanmean(strict)), float(np.nanmean(buffered))
