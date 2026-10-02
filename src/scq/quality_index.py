"""Survey-calibrated item quality index, and how often it reorders a
pairwise item comparison relative to an engagement-only ranking.

The "survey-calibrated" index is the empirical-Bayes (BLUP) shrinkage
estimate of each item's random effect from the crossed mixed-effects
model: `shrinkage * (item_mean_usefulness - grand_mean)`, where
`shrinkage = n_obs_per_item * var_item / (n_obs_per_item * var_item +
var_residual)`. In a fully crossed, balanced design (impressions.py),
this shrinkage factor is identical for every item, which means it is a
single monotonic rescaling of the raw item-mean usefulness rating and
therefore can NEVER change any pairwise ranking relative to using the raw
mean directly (see tests/test_quality_index.py). This is disclosed
plainly rather than presented as if the mixed model were doing ranking
work the raw mean could not: in THIS balanced design, the mixed model's
real contribution is estimating the variance components honestly (see
variance_decomp.py), not changing the item ranking. The reordering this
module measures is entirely attributable to the engagement metric
diverging from stated usefulness, which is exactly what the resume claim
and the README set out to measure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scq.variance_decomp import VarianceComponents


@dataclass
class QualityIndexResult:
    item_quality_index: pd.Series  # BLUP-shrunk survey-calibrated index, by item_id
    item_engagement_mean: pd.Series  # engagement-only ranking signal, by item_id
    shrinkage_factor: float


def compute_quality_index(
    df: pd.DataFrame,
    variance_components: VarianceComponents,
    value_col: str = "usefulness_rating",
    engagement_col: str = "engagement_score",
    item_col: str = "item_id",
) -> QualityIndexResult:
    item_means = df.groupby(item_col)[value_col].mean()
    n_obs_per_item = df.groupby(item_col).size()
    if n_obs_per_item.nunique() != 1:
        raise ValueError("compute_quality_index assumes a balanced design (equal obs per item)")
    n = int(n_obs_per_item.iloc[0])

    grand_mean = df[value_col].mean()
    denom = n * variance_components.var_item + variance_components.var_residual
    shrinkage = (n * variance_components.var_item / denom) if denom > 0 else 0.0

    quality_index = grand_mean + shrinkage * (item_means - grand_mean)
    engagement_mean = df.groupby(item_col)[engagement_col].mean()

    return QualityIndexResult(
        item_quality_index=quality_index,
        item_engagement_mean=engagement_mean,
        shrinkage_factor=float(shrinkage),
    )


def pairwise_reorder_rate(series_a: pd.Series, series_b: pd.Series) -> float:
    """Fraction of all unordered item pairs whose relative order under
    `series_a` disagrees with their relative order under `series_b`.
    Vectorized over the upper triangle of the full pairwise comparison
    matrix, not a sampled subset.
    """
    items = series_a.index
    a = series_a.to_numpy()
    b = series_b.reindex(items).to_numpy()
    n = len(items)
    i_idx, j_idx = np.triu_indices(n, k=1)
    order_a = a[i_idx] > a[j_idx]
    order_b = b[i_idx] > b[j_idx]
    return float((order_a != order_b).mean())
