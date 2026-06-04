"""Generate EDA plots for the data card. Run from the repo root: python docs/eda.py"""
# isort: skip_file
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config  # noqa: E402  pylint: disable=wrong-import-position
from data.load import load_data  # noqa: E402  pylint: disable=wrong-import-position

OUT = Path(__file__).resolve().parent / "img"


def main() -> None:
    """Save a target histogram and a scatter of the top numeric feature vs target."""
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    df = load_data(cfg)
    target = cfg.data.target
    feature = cfg.data.numeric_features[0]

    plt.figure()
    df[target].hist(bins=40)
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.title(f"{target} distribution")
    plt.tight_layout()
    plt.savefig(OUT / "target_hist.png")

    plt.figure()
    plt.scatter(df[feature], df[target], s=8, alpha=0.4)
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f"{feature} vs {target}")
    plt.tight_layout()
    plt.savefig(OUT / "feature_scatter.png")
    print(f"Saved plots to {OUT}")


if __name__ == "__main__":
    main()
