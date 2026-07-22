#!/usr/bin/env python3
"""Import-order-safe entry point for the 1D MMPD ordinal upscaling kill test."""

# MMPD vendors a top-level ``utils`` package. Import the repository utilities
# first so the MMPD Decoder can be added afterwards without shadowing the
# discriminator and dataset helpers.
from models.diffusion_tsf.ordinal_window_norm import ordinal_encode  # noqa: F401
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset  # noqa: F401
from utils.eval_discriminator_texture_staged_vs_mmpd import HorizonSliceDataset  # noqa: F401
from utils.eval_mmpd_gaussian_anchor import load_tsf_pack_pool  # noqa: F401

from experiments.ordinal_patch_refinement_killtest.mmpd_ordinal_upscale import main


if __name__ == "__main__":
    main()
