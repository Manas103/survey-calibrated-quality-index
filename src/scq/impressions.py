"""Simulated per-impression panel joining survey usefulness ratings to the
behavioral engagement log of the same session.

The design is fully crossed and balanced: every one of `n_respondents`
respondents rates every one of `n_items` items exactly once, so
`n_respondents * n_items` is the total impression count. This is a
deliberate simplification over a real intercept panel (see README, Honest
framing): it makes the crossed variance decomposition exactly checkable
against a closed-form two-way ANOVA (see variance_decomp.py), at the cost
of being a cleaner panel than any real survey-plus-behavioral-log join
would produce.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ImpressionConfig:
    n_respondents: int = 200
    n_items: int = 300
    item_effect_sd: float = 1.0
    respondent_effect_sd: float = 1.063  # chosen so Var(resp)/(Var(resp)+Var(item)+Var(resid)) = 0.38
    usefulness_residual_sd: float = 1.0
    engagement_quality_weight: float = 0.745  # weight on the SHARED usefulness signal
    engagement_specific_weight: float = 0.755  # weight on the engagement-only confound
    engagement_specific_sd: float = 1.0
    engagement_noise_sd: float = 0.5
    seed: int = 20260701


def simulate_impressions(config: ImpressionConfig = ImpressionConfig()) -> pd.DataFrame:
    """Build the fully-crossed impression panel.

    `usefulness_rating`: mu + item_effect + respondent_effect + residual.

    `engagement_score`: mu_e + engagement_quality_weight * item_effect +
    engagement_specific_weight * item_engagement_specific + noise. Every
    item has a SECOND, independent quality dimension,
    `item_engagement_specific` (a stand-in for whatever drives behavioral
    engagement but not stated usefulness, such as sensational or
    attention-grabbing content), uncorrelated with `item_effect` by
    construction. This is the real, structural reason engagement and
    stated usefulness diverge in this simulation, not merely sampling
    noise: averaging over many respondents reduces noise but does not
    reduce this structural divergence, which is why the item-level
    reordering claim (quality_index.py) needs this second dimension to be
    non-trivial at all (see README, "Why a second, engagement-specific
    item dimension is needed").
    """
    rng = np.random.default_rng(config.seed)
    item_effect = rng.normal(0.0, config.item_effect_sd, config.n_items)
    item_engagement_specific = rng.normal(0.0, config.engagement_specific_sd, config.n_items)
    respondent_effect = rng.normal(0.0, config.respondent_effect_sd, config.n_respondents)

    resp_idx = np.repeat(np.arange(config.n_respondents), config.n_items)
    item_idx = np.tile(np.arange(config.n_items), config.n_respondents)

    usefulness_noise = rng.normal(0.0, config.usefulness_residual_sd, len(resp_idx))
    engagement_noise = rng.normal(0.0, config.engagement_noise_sd, len(resp_idx))

    usefulness_rating = 3.0 + item_effect[item_idx] + respondent_effect[resp_idx] + usefulness_noise
    engagement_score = (
        50.0
        + config.engagement_quality_weight * item_effect[item_idx] * 10.0
        + config.engagement_specific_weight * item_engagement_specific[item_idx] * 10.0
        + engagement_noise * 10.0
    )

    return pd.DataFrame(
        {
            "respondent_id": resp_idx,
            "item_id": item_idx,
            "usefulness_rating": usefulness_rating,
            "engagement_score": engagement_score,
            "true_item_effect": item_effect[item_idx],
            "true_item_engagement_specific": item_engagement_specific[item_idx],
            "true_respondent_effect": respondent_effect[resp_idx],
        }
    )
