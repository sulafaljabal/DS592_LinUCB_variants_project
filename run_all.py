"""
run_all.py — Master script to run all experiments
===================================================
Run from the project root:
    python run_all.py

Or run individual experiments:
    python -m experiments.reproduce_paper
    python -m experiments.smooth_drift
    python -m experiments.sensitivity
"""

import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PLOT_DIR


def main():
    # Create output directory
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("╔══════════════════════════════════════════════════════╗")
    print("║   DS 592 Project: Weighted Linear Bandits           ║")
    print("║   for Non-Stationary Environments                   ║")
    print("║   Jelle Hendriks & Sulaf Al Jabal                   ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ─── Phase 1: Reproduce the paper ───
    print("\n\n▶ EXPERIMENT 1: Reproducing Russac et al. (2019)")
    from experiments.reproduce_paper import main as reproduce
    reproduce()

    # ─── Phase 2: Smooth drift experiments ───
    print("\n\n▶ EXPERIMENT 2: Smooth Drift Functions")
    from experiments.smooth_drift import main as smooth
    smooth()

    # ─── Phase 3: Sensitivity analysis ───
    print("\n\n▶ EXPERIMENT 3: Hyperparameter Sensitivity")
    from experiments.sensitivity import main as sensitivity
    sensitivity()

    print("\n\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Plots saved to: {PLOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
