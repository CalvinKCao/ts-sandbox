"""
Shape-Augmented iTransformer module.

This module provides the ShapeAugmentediTransformer which fuses:
1. iTransformer: Inverted transformer for time series
2. CNN Branch: Shape feature extraction from 2D representations
"""

from .shape_augmented_itransformer import (
    ShapeAugmentedConfig,
    ShapeAugmentediTransformer,
    VanillaiTransformer,
    CNNShapeEncoder,
    CrossAttentionFusion,
    count_parameters,
)

__all__ = [
    'ShapeAugmentedConfig',
    'ShapeAugmentediTransformer',
    'VanillaiTransformer',
    'CNNShapeEncoder',
    'CrossAttentionFusion',
    'count_parameters',
]

