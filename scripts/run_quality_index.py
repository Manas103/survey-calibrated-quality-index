"""Compute the survey-calibrated item quality index and measure how often
it reorders a pairwise item comparison relative to an engagement-only
ranking. Uses the fast closed-form ANOVA variance components (see
run_variance_decomp.py for the slower MixedLM cross-check of those same
variance components).

Usage:
    python scripts/run_quality_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scq.impressions import ImpressionConfig, simulate_impressions
from scq.quality_index import compute_quality_index, pairwise_reorder_rate
from scq.variance_decomp import anova_variance_components


def main() -> None:
    df = simulate_impressions(ImpressionConfig())
    vc = anova_variance_components(df)
    print(f"Variance components (ANOVA): var_respondent={vc.var_respondent:.4f}  var_item={vc.var_item:.4f}  var_residual={vc.var_residual:.4f}")

    result = compute_quality_index(df, vc)
    print(f"Shrinkage factor applied to every item's deviation from the grand mean: {result.shrinkage_factor:.4f}")

    reorder_rate = pairwise_reorder_rate(result.item_quality_index, result.item_engagement_mean)
    n_items = len(result.item_quality_index)
    n_pairs = n_items * (n_items - 1) // 2
    print(f"\n{n_items} items, {n_pairs} pairwise comparisons")
    print(f"PAIRWISE REORDER RATE (survey-calibrated index vs engagement-only ranking): {reorder_rate * 100:.2f}%")

    raw_mean = df.groupby("item_id")["usefulness_rating"].mean()
    sanity_reorder = pairwise_reorder_rate(result.item_quality_index, raw_mean)
    print(f"Sanity check, reorder rate vs the UNSHRUNK raw item mean (must be exactly 0): {sanity_reorder * 100:.4f}%")

    print("\nSUMMARY (resume claim vs measured):")
    print(
        json.dumps(
            {
                "n_items": n_items,
                "n_pairs": n_pairs,
                "claim_reorder_target": 0.27,
                "measured_reorder_rate": reorder_rate,
                "shrinkage_factor": result.shrinkage_factor,
                "sanity_reorder_vs_raw_mean": sanity_reorder,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
