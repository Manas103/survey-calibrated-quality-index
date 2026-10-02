"""Build the 60,000-impression panel and measure the Spearman correlation
between stated usefulness and behavioral engagement.

Usage:
    python scripts/run_impressions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import spearmanr

from scq.impressions import ImpressionConfig, simulate_impressions


def main() -> None:
    config = ImpressionConfig()
    df = simulate_impressions(config)
    print(f"n_respondents={config.n_respondents}, n_items={config.n_items}, n_impressions={len(df)}")

    rho, p = spearmanr(df["usefulness_rating"], df["engagement_score"])
    print(f"Spearman(usefulness_rating, engagement_score) across all impressions: {rho:.4f} (p={p:.2e})")

    print("\nSUMMARY (resume claim vs measured):")
    print(
        json.dumps(
            {
                "n_impressions": len(df),
                "claim_spearman_target": 0.31,
                "measured_spearman": float(rho),
                "p_value": float(p),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
