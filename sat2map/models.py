"""Generators and discriminator.

Five generator architectures share one interface, `make_generator(arch, size)`:

    unet             plain Pix2Pix U-Net (8 downsampling steps)
    unet_bottleneck  U-Net with a transformer block at the 1x1 bottleneck
    unet_mid         U-Net with a transformer block after e5 (8x8 at 256 input)
    resnet9          CycleGAN-style ResNet-9 generator
    attn_unet        Attention U-Net (gated skip connections)

`head="rgb"` gives a tanh image output (the image formulation). `head="seg"`
gives a three-class softmax output (background, road, water) on the same
backbone, used by the segmentation formulation.

The layers are created in the same order as in the original training
notebooks, so checkpoints saved there (`generator_best.weights.h5`) load into
these builders unchanged.
"""

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose,
                                     BatchNormalization, LeakyReLU, Activation,
                                     Dropout, Concatenate, Add, Multiply)
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model

ARCHS = ("unet", "unet_bottleneck", "unet_mid", "resnet9", "attn_unet")

# segmentation classes
N_CLASS, BG, ROAD, WATER = 3, 0, 1, 2


def _init():
    return RandomNormal(stddev=0.02)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(tf.keras.layers.Layer):
    """Self-attention over the spatial positions of a feature map.

    The feature map is flattened to h*w tokens of `embed_dim` channels, a
    learned positional embedding is added (attention is otherwise permutation
    invariant and could not use spatial layout), then one post-norm
    transformer layer is applied and the result reshaped back.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, pos_embed=True, **kw):
        super().__init__(**kw)
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.ff_dim, self.pos_embed = ff_dim, pos_embed
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)])
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.h, self.w = int(input_shape[1]), int(input_shape[2])
        if self.pos_embed:
            self.pos = self.add_weight(
                name="pos", shape=(1, self.h * self.w, self.embed_dim),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        x = tf.reshape(inputs, (-1, self.h * self.w, self.embed_dim))
        if self.pos_embed:
            x = x + self.pos
        x = self.ln1(self.att(x, x) + x)
        x = self.ln2(self.ffn(x) + x)
        return tf.reshape(x, (-1, self.h, self.w, self.embed_dim))

    def get_config(self):
        c = super().get_config()
        c.update(embed_dim=self.embed_dim, num_heads=self.num_heads,
                 ff_dim=self.ff_dim, pos_embed=self.pos_embed)
        return c


# ---------------------------------------------------------------------------
# U-Net (optionally with a transformer block)
# ---------------------------------------------------------------------------

def enc_block(x, n, batchnorm=True):
    g = Conv2D(n, (4, 4), strides=2, padding="same", kernel_initializer=_init())(x)
    if batchnorm:
        g = BatchNormalization()(g)
    return LeakyReLU(negative_slope=0.2)(g)


def dec_block(x, skip, n, dropout=True):
    g = Conv2DTranspose(n, (4, 4), strides=2, padding="same",
                        kernel_initializer=_init())(x)
    g = BatchNormalization()(g)
    if dropout:
        g = Dropout(0.5)(g)
    return Activation("relu")(Concatenate()([g, skip]))


def build_unet(size, placement=None, num_heads=4, ff_dim=1024, pos_embed=True,
               out_act="tanh"):
    """Pix2Pix U-Net. `placement` is None, "bottleneck" or "mid".

    At 256 input the bottleneck is 1x1, where attention over a single token
    has nothing to relate. "mid" places the block after e5 instead, which is
    8x8 at 256 input and 16x16 at 512.
    """
    init = _init()
    inp = Input(shape=(size, size, 3))

    e1 = enc_block(inp, 64, batchnorm=False)
    e2 = enc_block(e1, 128)
    e3 = enc_block(e2, 256)
    e4 = enc_block(e3, 512)
    e5 = enc_block(e4, 512)
    if placement == "mid":
        e5 = TransformerBlock(512, num_heads, ff_dim, pos_embed,
                              name="transformer_mid")(e5)
    e6 = enc_block(e5, 512)
    e7 = enc_block(e6, 512)

    b = Conv2D(512, (4, 4), strides=2, padding="same", kernel_initializer=init)(e7)
    if placement == "bottleneck":
        b = TransformerBlock(512, num_heads, ff_dim, pos_embed,
                             name="transformer_bottleneck")(b)
    b = Activation("relu")(b)

    d1 = dec_block(b,  e7, 512)
    d2 = dec_block(d1, e6, 512)
    d3 = dec_block(d2, e5, 512)
    d4 = dec_block(d3, e4, 512, dropout=False)
    d5 = dec_block(d4, e3, 256, dropout=False)
    d6 = dec_block(d5, e2, 128, dropout=False)
    d7 = dec_block(d6, e1, 64,  dropout=False)

    out = Conv2DTranspose(3, (4, 4), strides=2, padding="same",
                          kernel_initializer=init)(d7)
    return Model(inp, Activation(out_act)(out), name="unet")


# ---------------------------------------------------------------------------
# ResNet-9 (CycleGAN / Johnson et al.)
# ---------------------------------------------------------------------------

def resnet_block(x, n):
    init = _init()
    g = Conv2D(n, (3, 3), padding="same", kernel_initializer=init)(x)
    g = BatchNormalization()(g)
    g = Activation("relu")(g)
    g = Conv2D(n, (3, 3), padding="same", kernel_initializer=init)(g)
    g = BatchNormalization()(g)
    return Add()([g, x])


def build_resnet9(size, out_act="tanh", n_blocks=9, base=64):
    init = _init()
    inp = Input(shape=(size, size, 3))

    g = Conv2D(base, (7, 7), padding="same", kernel_initializer=init)(inp)
    g = BatchNormalization()(g)
    g = Activation("relu")(g)

    for n in (base * 2, base * 4):                  # two downsampling steps
        g = Conv2D(n, (3, 3), strides=2, padding="same", kernel_initializer=init)(g)
        g = BatchNormalization()(g)
        g = Activation("relu")(g)

    for _ in range(n_blocks):                       # residual blocks at size/4
        g = resnet_block(g, base * 4)

    for n in (base * 2, base):                      # two upsampling steps
        g = Conv2DTranspose(n, (3, 3), strides=2, padding="same",
                            kernel_initializer=init)(g)
        g = BatchNormalization()(g)
        g = Activation("relu")(g)

    out = Conv2D(3, (7, 7), padding="same", kernel_initializer=init)(g)
    return Model(inp, Activation(out_act)(out), name="resnet9")


# ---------------------------------------------------------------------------
# Attention U-Net (Oktay et al.)
# ---------------------------------------------------------------------------

def attention_gate(skip, gating, inter):
    """Gate the encoder skip using the coarser decoder feature one level down."""
    init = _init()
    g = Conv2DTranspose(inter, (4, 4), strides=2, padding="same",
                        kernel_initializer=init)(gating)
    x = Conv2D(inter, (1, 1), padding="same", kernel_initializer=init)(skip)
    a = Activation("relu")(Add()([g, x]))
    a = Conv2D(1, (1, 1), padding="same", kernel_initializer=init)(a)
    a = Activation("sigmoid")(a)
    return Multiply()([skip, a])


def dec_block_attn(x, skip, n, inter, dropout=True):
    gated = attention_gate(skip, x, inter)
    g = Conv2DTranspose(n, (4, 4), strides=2, padding="same",
                        kernel_initializer=_init())(x)
    g = BatchNormalization()(g)
    if dropout:
        g = Dropout(0.5)(g)
    return Activation("relu")(Concatenate()([g, gated]))


def build_attn_unet(size, out_act="tanh"):
    init = _init()
    inp = Input(shape=(size, size, 3))

    e1 = enc_block(inp, 64, batchnorm=False)
    e2 = enc_block(e1, 128)
    e3 = enc_block(e2, 256)
    e4 = enc_block(e3, 512)
    e5 = enc_block(e4, 512)
    e6 = enc_block(e5, 512)
    e7 = enc_block(e6, 512)

    b = Conv2D(512, (4, 4), strides=2, padding="same", kernel_initializer=init)(e7)
    b = Activation("relu")(b)

    d1 = dec_block_attn(b,  e7, 512, 512)
    d2 = dec_block_attn(d1, e6, 512, 512)
    d3 = dec_block_attn(d2, e5, 512, 512)
    d4 = dec_block_attn(d3, e4, 512, 256, dropout=False)
    d5 = dec_block_attn(d4, e3, 256, 128, dropout=False)
    d6 = dec_block_attn(d5, e2, 128, 64,  dropout=False)
    d7 = dec_block_attn(d6, e1, 64,  32,  dropout=False)

    out = Conv2DTranspose(3, (4, 4), strides=2, padding="same",
                          kernel_initializer=init)(d7)
    return Model(inp, Activation(out_act)(out), name="attn_unet")


# ---------------------------------------------------------------------------
# selector
# ---------------------------------------------------------------------------

def make_generator(arch, size, head="rgb", num_heads=4, ff_dim=1024,
                   pos_embed=True):
    """Build one of the five generators.

    head: "rgb" for a tanh image output, "seg" for a 3-class softmax output.
    The two heads have identical weights topology, so the head must be known
    from the run, not inferred from the checkpoint.
    """
    act = "softmax" if head == "seg" else "tanh"
    if arch == "unet":
        return build_unet(size, None, num_heads, ff_dim, pos_embed, act)
    if arch == "unet_bottleneck":
        return build_unet(size, "bottleneck", num_heads, ff_dim, pos_embed, act)
    if arch == "unet_mid":
        return build_unet(size, "mid", num_heads, ff_dim, pos_embed, act)
    if arch == "resnet9":
        return build_resnet9(size, act)
    if arch == "attn_unet":
        return build_attn_unet(size, act)
    raise ValueError("unknown arch %r, expected one of %s" % (arch, ARCHS))


# ---------------------------------------------------------------------------
# Discriminator (70x70 PatchGAN)
# ---------------------------------------------------------------------------

def build_discriminator(size):
    init = _init()
    src, tar = Input(shape=(size, size, 3)), Input(shape=(size, size, 3))
    x = Concatenate()([src, tar])
    x = LeakyReLU(negative_slope=0.2)(
        Conv2D(64, (4, 4), strides=2, padding="same", kernel_initializer=init)(x))
    for n in (128, 256, 512):
        x = Conv2D(n, (4, 4), strides=2, padding="same", kernel_initializer=init)(x)
        x = LeakyReLU(negative_slope=0.2)(BatchNormalization()(x))
    x = Conv2D(512, (4, 4), padding="same", kernel_initializer=init)(x)
    x = LeakyReLU(negative_slope=0.2)(BatchNormalization()(x))
    out = Conv2D(1, (4, 4), padding="same", kernel_initializer=init)(x)
    return Model([src, tar], out, name="discriminator")
