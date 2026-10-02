"""Fit the crossed variance decomposition two ways (closed-form ANOVA and
REML MixedLM) on the full 60,000-impression panel and report the percent
of usefulness-rating variance at the respondent level.

The MixedLM fit on the full panel (200 respondents crossed with 300
items) takes roughly 2-3 minutes on this machine; this is noted up front
so running this script is not mistaken for hanging.

Usage:
    python scripts/run_variance_decomp.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scq.impressions import ImpressionConfig, simulate_impressions
from scq.variance_decomp import anova_variance_components, fit_mixedlm_variance_components


def main() -> None:
    df = simulate_impressions(ImpressionConfig())
    print(f"n_impressions={len(df)}")

    t0 = time.time()
    vc_anova = anova_variance_components(df)
    print(f"\nANOVA (closed-form, reference oracle) [{time.time() - t0:.2f}s]:")
    print(f"  var_respondent={vc_anova.var_respondent:.4f}  var_item={vc_anova.var_item:.4f}  var_residual={vc_anova.var_residual:.4f}")
    print(f"  pct_at_respondent={vc_anova.pct_at_respondent * 100:.2f}%")

    print("\nFitting MixedLM (REML, crossed random effects)... this takes a few minutes.")
    t0 = time.time()
    vc_mixedlm = fit_mixedlm_variance_components(df)
    print(f"MixedLM [{time.time() - t0:.2f}s]:")
    print(f"  var_respondent={vc_mixedlm.var_respondent:.4f}  var_item={vc_mixedlm.var_item:.4f}  var_residual={vc_mixedlm.var_residual:.4f}")
    print(f"  pct_at_respondent={vc_mixedlm.pct_at_respondent * 100:.2f}%")

    print("\nSUMMARY (resume claim vs measured):")
    print(
        json.dumps(
            {
                "claim_pct_at_respondent_target": 0.38,
                "measured_pct_at_respondent_anova": vc_anova.pct_at_respondent,
                "measured_pct_at_respondent_mixedlm": vc_mixedlm.pct_at_respondent,
                "var_respondent_anova": vc_anova.var_respondent,
                "var_item_anova": vc_anova.var_item,
                "var_residual_anova": vc_anova.var_residual,
                "var_respondent_mixedlm": vc_mixedlm.var_respondent,
                "var_item_mixedlm": vc_mixedlm.var_item,
                "var_residual_mixedlm": vc_mixedlm.var_residual,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
