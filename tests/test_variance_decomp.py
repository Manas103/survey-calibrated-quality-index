import numpy as np
import pandas as pd
import pytest

from scq.impressions import ImpressionConfig, simulate_impressions
from scq.variance_decomp import anova_variance_components, fit_mixedlm_variance_components


def _toy_balanced_panel(seed=0, R=8, K=10, resp_sd=2.0, item_sd=1.0, resid_sd=0.5):
    rng = np.random.default_rng(seed)
    resp_eff = rng.normal(0, resp_sd, R)
    item_eff = rng.normal(0, item_sd, K)
    resp_idx = np.repeat(np.arange(R), K)
    item_idx = np.tile(np.arange(K), R)
    resid = rng.normal(0, resid_sd, R * K)
    y = 10.0 + resp_eff[resp_idx] + item_eff[item_idx] + resid
    return pd.DataFrame({"usefulness_rating": y, "respondent_id": resp_idx, "item_id": item_idx})


def test_anova_oracle_recovers_known_variances_on_a_large_toy_panel():
    """With enough respondents and items, the method-of-moments ANOVA
    estimator must recover the TRUE generating variances closely (this is
    an unbiased estimator, so the only error here is finite-sample noise,
    not approximation error).
    """
    df = _toy_balanced_panel(seed=1, R=200, K=300, resp_sd=1.5, item_sd=1.0, resid_sd=1.0)
    vc = anova_variance_components(df)
    # Tolerances are set from the estimator's own known sampling
    # variability (a method-of-moments variance estimate from R-1=199 or
    # K-1=299 "degrees of freedom" has a relative standard error of
    # roughly sqrt(2/df), about 10-14% here), not an arbitrary margin.
    assert abs(vc.var_respondent - 1.5**2) < 0.45
    assert abs(vc.var_item - 1.0**2) < 0.25
    assert abs(vc.var_residual - 1.0**2) < 0.05


def test_anova_requires_balanced_design():
    df = pd.DataFrame(
        {
            "usefulness_rating": [1.0, 2.0, 3.0],
            "respondent_id": [0, 0, 1],
            "item_id": [0, 1, 0],
        }
    )
    with pytest.raises(ValueError):
        anova_variance_components(df)


def test_variances_are_never_negative():
    for seed in range(5):
        df = _toy_balanced_panel(seed=seed, R=10, K=12, resp_sd=0.01, item_sd=0.01, resid_sd=1.0)
        vc = anova_variance_components(df)
        assert vc.var_respondent >= 0.0
        assert vc.var_item >= 0.0
        assert vc.var_residual >= 0.0


def test_mixedlm_matches_anova_oracle_on_a_small_toy_panel():
    """Reference-oracle diff: a small, fast-to-fit toy panel (24
    respondents x 30 items = 720 rows) is run through BOTH estimators;
    they must agree closely, since the ANOVA estimator's correctness
    follows directly from textbook expected-mean-square identities for
    this exact balanced design.
    """
    df = _toy_balanced_panel(seed=2, R=24, K=30, resp_sd=1.3, item_sd=0.9, resid_sd=1.0)
    vc_anova = anova_variance_components(df)
    vc_mixedlm = fit_mixedlm_variance_components(df)

    assert abs(vc_anova.var_respondent - vc_mixedlm.var_respondent) < 0.3
    assert abs(vc_anova.var_item - vc_mixedlm.var_item) < 0.3
    assert abs(vc_anova.var_residual - vc_mixedlm.var_residual) < 0.2


def test_pct_at_respondent_sums_consistently():
    vc = anova_variance_components(_toy_balanced_panel(seed=3))
    assert 0.0 <= vc.pct_at_respondent <= 1.0
    assert 0.0 <= vc.pct_at_item <= 1.0
    assert vc.pct_at_respondent + vc.pct_at_item <= 1.0 + 1e-9
