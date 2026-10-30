"""Run names and checkpoint discovery.

Every training run writes to a folder named after its configuration and saves
`generator_best.weights.h5` there. The evaluation notebooks recover the
architecture, output head, input size, loss weights and seed from that name.

Current format (written by notebooks/01_train.ipynb):

    unet_mid_res256_adv03_rw10_n2400_s96
    ^arch    ^size  ^adv   ^road ^n   ^seed % 1000
             weight x10  weight

Segmentation runs (notebooks/01b_train_segmentation.ipynb) carry a `seg_`
prefix, e.g. `seg_unet_res256_dice10_ce10_adv00_hflip_n2400_s96`. Older image
runs named `res256_<none|bottleneck|mid>_adv03_...` still parse.
"""

import re
from pathlib import Path


def run_name(arch, size, adv_weight, road_weight, n_train, seed):
    return "%s_res%d_adv%02d_rw%02d_n%d_s%d" % (
        arch, size, int(round(adv_weight * 10)), int(road_weight), n_train,
        seed % 1000)


def seg_run_name(arch, size, dice_weight, ce_weight, adv_weight, aug,
                 n_train, seed):
    return "seg_%s_res%d_dice%02d_ce%02d_adv%02d_%s_n%d_s%d" % (
        arch, size, int(round(dice_weight * 10)), int(round(ce_weight * 10)),
        int(round(adv_weight * 10)), aug, n_train, seed % 1000)


def head_of(name):
    return "seg" if name.lower().startswith("seg_") else "rgb"


def arch_of(name):
    n = name.lower()
    if n.startswith("seg_"):
        n = n[4:]
    if n.startswith("attn_unet"):
        return "attn_unet"
    if n.startswith("resnet9"):
        return "resnet9"
    if "bottleneck" in n:
        return "unet_bottleneck"
    if "mid" in n:
        return "unet_mid"
    return "unet"


def _num(pattern, name, default=None):
    m = re.search(pattern, name)
    return int(m.group(1)) if m else default


def parse(name):
    """dict(arch, head, size, adv, rw, seed) from a run name."""
    return dict(
        arch=arch_of(name),
        head=head_of(name),
        size=_num(r"res(\d+)", name, 256),
        adv=_num(r"adv(\d+)", name, 0) / 10.0,
        rw=_num(r"rw(\d+)", name),
        seed=str(_num(r"_s(\d+)$", name, "")),
    )


def label_of(arch, head):
    return arch if head == "rgb" else arch + "+seg"


def find_checkpoints(root, filename="generator_best.weights.h5"):
    """{run_name: path} for every checkpoint found anywhere under root."""
    found = {}
    for p in sorted(Path(root).rglob(filename)):
        found[p.parent.name] = str(p)
    return found
