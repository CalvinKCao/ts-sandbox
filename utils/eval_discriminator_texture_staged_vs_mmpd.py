"""Shim: renamed to ``utils.disc_shared``.

Kept so existing imports keep working. Prefer importing ``utils.disc_shared`` directly.
"""

from utils.disc_shared import *  # noqa: F401,F403
from utils.disc_shared import (  # noqa: F401
    DEFAULT_DISC_OUTPUT,
    DISC_ARCH_CHOICES,
    FAKE_SOURCES,
    HorizonSliceDataset,
    InvertedSliceDiscriminator,
    LOG2,
    apply_disc_pack_protocol,
    apply_smoke_defaults,
    binary_auroc,
    binary_mmpd_train_scaler_map,
    build_raw_bundle,
    build_slice_discriminator,
    collect_partials,
    evaluate_classifier,
    load_ordinal_ladder_for_run,
    parse_args,
    split_windows,
    stable_hash,
    train_classifier,
    window_level_metrics,
    write_json,
    zscore_time,
)
