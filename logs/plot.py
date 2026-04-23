#!/usr/bin/env python3
"""Plot training curves from JSON logs.

Usage:
    python logs/plot.py                         # plot all available logs
    python logs/plot.py q dyna-q-linear         # plot specific algorithms
    python logs/plot.py --window 100            # change smoothing window
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(LOG_DIR, "comparison.png")

COLORS = ["green", "blue", "orange", "red", "purple", "brown"]


def moving_avg(x, window=50):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def load_rewards(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("rewards", [])


def find_logs(names=None):
    """Return list of (algo_name, json_path) sorted by name."""
    if names:
        results = []
        for name in names:
            path = os.path.join(LOG_DIR, f"{name}.json")
            if os.path.exists(path):
                results.append((name, path))
            else:
                print(f"Warning: {path} not found, skipping")
        return results

    paths = sorted(glob.glob(os.path.join(LOG_DIR, "*.json")))
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algos", nargs="*", help="algorithm names to plot (default: all)")
    parser.add_argument("--window", type=int, default=50, help="smoothing window size")
    args = parser.parse_args()

    logs = find_logs(args.algos if args.algos else None)
    if not logs:
        print("No JSON logs found in logs/")
        sys.exit(1)

    # Per-algorithm individual plots
    for i, (name, path) in enumerate(logs):
        rewards = load_rewards(path)
        if not rewards:
            continue
        smoothed = moving_avg(rewards, args.window)
        color = COLORS[i % len(COLORS)]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rewards, alpha=0.15, color=color)
        ax.plot(smoothed, color=color, label=f"avg-{args.window}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title(f"{name} ({len(rewards)} episodes)")
        ax.legend()
        ax.grid()
        fig.tight_layout()
        individual_path = os.path.join(LOG_DIR, f"{name}_curve.png")
        fig.savefig(individual_path, dpi=120)
        plt.close(fig)
        print(f"Saved → {individual_path}")

    # Combined comparison plot
    if len(logs) > 1:
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, (name, path) in enumerate(logs):
            rewards = load_rewards(path)
            if not rewards:
                continue
            smoothed = moving_avg(rewards, args.window)
            color = COLORS[i % len(COLORS)]
            ax.plot(rewards, alpha=0.15, color=color)
            ax.plot(smoothed, color=color, label=f"{name} ({len(rewards)} ep)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Training Comparison")
        ax.legend()
        ax.grid()
        fig.tight_layout()
        fig.savefig(OUT_PATH, dpi=120)
        plt.close(fig)
        print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
