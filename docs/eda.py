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
    """Save a target histogram and a distance-vs-time scatter."""
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    df = load_data(cfg)

    plt.figure()
    df[cfg.data.target].hist(bins=30)
    plt.xlabel("Delivery time (min)")
    plt.ylabel("Count")
    plt.title("Delivery time distribution")
    plt.tight_layout()
    plt.savefig(OUT / "target_hist.png")

    plt.figure()
    plt.scatter(df["Distance_km"], df[cfg.data.target], s=8, alpha=0.4)
    plt.xlabel("Distance (km)")
    plt.ylabel("Delivery time (min)")
    plt.title("Distance vs delivery time")
    plt.tight_layout()
    plt.savefig(OUT / "distance_vs_time.png")
    print(f"Saved plots to {OUT}")


if __name__ == "__main__":
    main()
