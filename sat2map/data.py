"""Input pipeline.

Each dataset file is one concatenated PNG: satellite on the left, styled map
on the right. Loading splits it at the midpoint of its width, so it works at
whatever resolution the files were stored (the collected set is 1024x512),
resizes each half to `size` with area resampling and scales to [-1, 1].

Expected layout:

    <DATA_ROOT>/train/*.png   2400 pairs
    <DATA_ROOT>/val/*.png      300 pairs
    <DATA_ROOT>/test/*.png     300 pairs
"""

from pathlib import Path

import tensorflow as tf


def file_list(data_root, split):
    """Sorted PNG paths of one split."""
    return sorted(str(p) for p in (Path(data_root) / split).glob("*.png"))


def make_loader(size):
    def load_pair(path):
        img = tf.cast(tf.io.decode_png(tf.io.read_file(path), channels=3),
                      tf.float32)
        w = tf.shape(img)[1] // 2
        src = tf.image.resize(img[:, :w, :], [size, size], method="area")
        tar = tf.image.resize(img[:, w:, :], [size, size], method="area")
        return src / 127.5 - 1.0, tar / 127.5 - 1.0
    return load_pair


def augment(src, tar):
    """Horizontal flip with probability 0.5, applied to both halves."""
    if tf.random.uniform(()) > 0.5:
        src = tf.image.flip_left_right(src)
        tar = tf.image.flip_left_right(tar)
    return src, tar


def augment_d4(src, tar):
    """One of the eight rotations/reflections of the square, applied to both.

    Optional (aug="d4"); every thesis run used the horizontal flip only.
    """
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    src, tar = tf.image.rot90(src, k), tf.image.rot90(tar, k)
    return augment(src, tar)


def make_ds(files, size, batch=1, training=False, seed=None, aug="hflip"):
    """tf.data pipeline. Training shuffles every epoch and augments
    (aug="hflip" as in the thesis, or "d4")."""
    ds = tf.data.Dataset.from_tensor_slices(files)
    if training:
        ds = ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(make_loader(size), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        fn = augment_d4 if aug == "d4" else augment
        ds = ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch if training else 1).prefetch(tf.data.AUTOTUNE)


def road_mask_tf(img_pm1, thresh):
    """Road mask of a [-1, 1] map image.

    The collection styling draws roads in pure white, so a pixel is road when
    its darkest channel, rescaled to [0, 1], is at least `thresh`.
    """
    return tf.reduce_min(img_pm1 * 0.5 + 0.5, axis=-1) >= thresh
