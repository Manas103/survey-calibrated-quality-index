"""Crossed random-effects variance decomposition: how much of the variance
in usefulness ratings sits at the respondent level versus the item level
versus unexplained residual.

Two independent estimators are implemented and cross-checked against each
other (see tests/test_variance_decomp.py and README, Validation):
`fit_mixedlm_variance_components`, a REML fit via
`statsmodels.regression.mixed_linear_model.MixedLM` with item modeled as a
variance component crossed with the respondent grouping factor, and
`anova_variance_components`, a closed-form method-of-moments estimator
using the classical two-way balanced random-effects ANOVA identities. The
ANOVA estimator is exact only because the impression panel is fully
crossed and balanced (see impressions.py); it is the reference oracle for
the REML fit, not the other way around, because its correctness follows
directly from textbook expected-mean-square algebra rather than from
trusting a third-party optimizer's convergence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass(frozen=True)
class VarianceComponents:
    var_respondent: float
    var_item: float
    var_residual: float

    @property
    def pct_at_respondent(self) -> float:
        total = self.var_respondent + self.var_item + self.var_residual
        return float(self.var_respondent / total) if total > 0 else 0.0

    @property
    def pct_at_item(self) -> float:
        total = self.var_respondent + self.var_item + self.var_residual
        return float(self.var_item / total) if total > 0 else 0.0


def anova_variance_components(
    df: pd.DataFrame,
    value_col: str = "usefulness_rating",
    respondent_col: str = "respondent_id",
    item_col: str = "item_id",
) -> VarianceComponents:
    """Closed-form method-of-moments variance components for a FULLY
    CROSSED, BALANCED two-way random-effects design (every respondent
    rates every item exactly once, no interaction term, since there is
    only one observation per respondent-item cell).

    Classic identities (Searle, Casella & McCulloch, "Variance
    Components", ch. 4): with R respondents and K items,
        MS_respondent = K * sum_r (ybar_r. - ybar..)^2 / (R-1)
        MS_item       = R * sum_k (ybar_.k - ybar..)^2 / (K-1)
        MS_error      = SS_total - SS_respondent - SS_item, over (R-1)(K-1) df
        E[MS_respondent] = sigma_e^2 + K * sigma_r^2
        E[MS_item]       = sigma_e^2 + R * sigma_item^2
        E[MS_error]      = sigma_e^2
    """
    grand_mean = df[value_col].mean()
    resp_means = df.groupby(respondent_col)[value_col].mean()
    item_means = df.groupby(item_col)[value_col].mean()
    R = len(resp_means)
    K = len(item_means)
    n = len(df)
    if n != R * K:
        raise ValueError("anova_variance_components requires a fully crossed, balanced design")

    ss_respondent = K * ((resp_means - grand_mean) ** 2).sum()
    ss_item = R * ((item_means - grand_mean) ** 2).sum()
    ss_total = ((df[value_col] - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_respondent - ss_item

    ms_respondent = ss_respondent / (R - 1)
    ms_item = ss_item / (K - 1)
    ms_error = ss_error / ((R - 1) * (K - 1))

    var_residual = ms_error
    var_respondent = max((ms_respondent - ms_error) / K, 0.0)
    var_item = max((ms_item - ms_error) / R, 0.0)

    return VarianceComponents(
        var_respondent=float(var_respondent),
        var_item=float(var_item),
        var_residual=float(var_residual),
    )


def fit_mixedlm_variance_components(
    df: pd.DataFrame,
    value_col: str = "usefulness_rating",
    respondent_col: str = "respondent_id",
    item_col: str = "item_id",
) -> VarianceComponents:
    """REML fit of a crossed random-effects model via statsmodels
    MixedLM: respondent as the `groups` random intercept, item as a
    `vc_formula` variance component crossed with respondent. `re_formula`
    is passed EXPLICITLY as "1": the first version of this function
    omitted it and relied on the default, which silently drops the
    respondent-level random intercept entirely once a vc_formula is also
    supplied (see README, "What broke").
    """
    work = df[[value_col, respondent_col, item_col]].copy()
    work[respondent_col] = work[respondent_col].astype(str)
    work[item_col] = work[item_col].astype(str)

    model = smf.mixedlm(
        f"{value_col} ~ 1",
        work,
        groups=work[respondent_col],
        re_formula="1",
        vc_formula={"item": f"0 + C({item_col})"},
    )
    result = model.fit(reml=True)

    var_respondent = float(result.cov_re.iloc[0, 0])
    var_item = float(result.vcomp[0])
    var_residual = float(result.scale)

    return VarianceComponents(
        var_respondent=var_respondent, var_item=var_item, var_residual=var_residual
    )
