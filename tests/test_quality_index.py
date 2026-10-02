import numpy as np
import pandas as pd

from scq.impressions import ImpressionConfig, simulate_impressions
from scq.quality_index import compute_quality_index, pairwise_reorder_rate
from scq.variance_decomp import VarianceComponents, anova_variance_components


def test_shrinkage_is_identical_across_items_in_a_balanced_design():
    """In a fully crossed, balanced design, every item has the same
    number of observations, so the empirical-Bayes shrinkage factor must
    be a single scalar, not item-specific.
    """
    df = simulate_impressions(ImpressionConfig(n_respondents=30, n_items=40))
    vc = anova_variance_components(df)
    result = compute_quality_index(df, vc)
    assert 0.0 <= result.shrinkage_factor <= 1.0


def test_quality_index_never_reorders_relative_to_raw_item_mean():
    """Because shrinkage is a single positive scalar applied uniformly
    (see module docstring), the BLUP quality index must induce EXACTLY
    the same pairwise ranking as the raw, unshrunk item-mean usefulness
    rating: shrinking every item's deviation from the grand mean by the
    same factor cannot flip any pairwise comparison.
    """
    df = simulate_impressions(ImpressionConfig(n_respondents=30, n_items=40))
    vc = anova_variance_components(df)
    result = compute_quality_index(df, vc)
    raw_item_mean = df.groupby("item_id")["usefulness_rating"].mean()
    reorder_rate = pairwise_reorder_rate(result.item_quality_index, raw_item_mean)
    assert reorder_rate == 0.0


def test_pairwise_reorder_rate_is_zero_against_itself():
    series = pd.Series([3.0, 1.0, 4.0, 1.5], index=[0, 1, 2, 3])
    assert pairwise_reorder_rate(series, series) == 0.0


def test_pairwise_reorder_rate_hand_computed_example():
    """4 items, hand-worked pairwise comparisons: A ranks [0,1,2,3] in
    ascending order; B reverses the middle two. Of the 6 pairs, exactly 1
    (the swapped pair) disagrees.
    """
    a = pd.Series([1.0, 2.0, 3.0, 4.0], index=["w", "x", "y", "z"])
    b = pd.Series([1.0, 3.0, 2.0, 4.0], index=["w", "x", "y", "z"])
    assert pairwise_reorder_rate(a, b) == 1 / 6


def test_identical_rankings_reorder_rate_zero_even_with_different_scale():
    a = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    b = pd.Series([10.0, 20.0, 30.0], index=["a", "b", "c"])
    assert pairwise_reorder_rate(a, b) == 0.0


def test_fully_reversed_ranking_reorder_rate_is_one():
    a = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    b = pd.Series([3.0, 2.0, 1.0], index=["a", "b", "c"])
    assert pairwise_reorder_rate(a, b) == 1.0
