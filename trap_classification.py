import numpy as np

# Per-block group status: on the good target, on the bad target, or on neither
TRAP_CLASSES = ["good", "bad", "lost"]


def block_status(occ_good, occ_bad):
    """[..., Islands] occupancy fractions -> int8 status (index into TRAP_CLASSES).
    A group is "on" a target if its median member is within the radius for >50% of the block."""
    status = np.full(occ_good.shape, 2, dtype=np.int8)
    status[occ_bad > 0.5] = 1
    status[occ_good > 0.5] = 0
    return status


def classify_window(status):
    """[Blocks, Islands] status -> per-group class over the window.
    Holding (0) if on the good target in most blocks. Otherwise trapped, either on the
    bad target (1) or on neither (2), whichever is more frequent (ties go to bad)."""
    on_good = (status == 0).mean(0)
    on_bad = (status == 1).mean(0)
    lost = (status == 2).mean(0)
    cls = np.where(on_bad >= lost, 1, 2).astype(np.int8)
    cls[on_good > 0.5] = 0
    return cls
