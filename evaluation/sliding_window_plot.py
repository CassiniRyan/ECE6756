#!/usr/bin/env python3
"""Plot sliding-window average reward curves from JSON training logs.

Examples:
    python evaluation/sliding_window_plot.py --log-dir logs
    python evaluation/sliding_window_plot.py --log-dir logs --tasks q rwm-q
    python evaluation/sliding_window_plot.py --log-dir logs --window 100
    python evaluation/sliding_window_plot.py --log-dir logs --output evaluation/reward_window.png
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


DEFAULT_WINDOW = 100

TASK_COLORS = {
    "q": "tab:blue",
    "dyna-q": "tab:green",
    "dyna-q-discret": "tab:green",
    "dyna-q-linear": "tab:orange",
    "rwm-q": "tab:red",
    "rwm-predict": "tab:purple",
}

FALLBACK_COLORS = [
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def sliding_average(values, window):
    """Return a valid-mode sliding average and matching episode numbers."""
    rewards = np.array(values, dtype=np.float32)
    if rewards.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    if window <= 1:
        episodes = np.arange(1, rewards.size + 1)
        return episodes, rewards
    if rewards.size < window:
        episodes = np.arange(1, rewards.size + 1)
        return episodes, rewards

    kernel = np.ones(window, dtype=np.float32) / float(window)
    averaged = np.convolve(rewards, kernel, mode="valid")
    episodes = np.arange(window, rewards.size + 1)
    return episodes, averaged


def sliding_envelope(values, window, mode):
    """Return a sliding upper/lower envelope and matching episode numbers."""
    rewards = np.array(values, dtype=np.float32)
    if rewards.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    if window <= 1 or rewards.size < window:
        episodes = np.arange(1, rewards.size + 1)
        return episodes, rewards

    windows = np.lib.stride_tricks.sliding_window_view(rewards, window_shape=window)
    if mode == "upper":
        envelope = np.max(windows, axis=1)
    elif mode == "lower":
        envelope = np.min(windows, axis=1)
    else:
        raise ValueError(f"Unsupported envelope mode: {mode}")
    episodes = np.arange(window, rewards.size + 1)
    return episodes, envelope


def load_rewards(path):
    with open(path, "r") as f:
        data = json.load(f)
    rewards = data.get("rewards", [])
    if not isinstance(rewards, list):
        raise ValueError(f"{path} has no list-valued 'rewards' field")
    return rewards


def find_logs(log_dir, tasks=None):
    """Return sorted (task_name, json_path) pairs."""
    if tasks:
        logs = []
        for task in tasks:
            path = os.path.join(log_dir, f"{task}.json")
            if os.path.exists(path):
                logs.append((task, path))
            else:
                print(f"Warning: missing log for task '{task}': {path}")
        return logs

    paths = sorted(glob.glob(os.path.join(log_dir, "*.json")))
    return [(os.path.splitext(os.path.basename(path))[0], path) for path in paths]


def color_for(task, fallback_index):
    if task in TASK_COLORS:
        return TASK_COLORS[task]
    return FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]


def output_paths(base_output):
    """Return average, upper-envelope, and lower-envelope output paths."""
    root, ext = os.path.splitext(base_output)
    if not ext:
        ext = ".png"
    return {
        "average": f"{root}{ext}",
        "upper": f"{root}_upper_envelope{ext}",
        "lower": f"{root}_lower_envelope{ext}",
    }


def plot_reward_summary(logs, window, output, mode):
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = 0

    for i, (task, path) in enumerate(logs):
        rewards = load_rewards(path)
        if not rewards:
            print(f"Warning: skipping empty rewards log: {path}")
            continue

        raw_episodes = np.arange(1, len(rewards) + 1)
        color = color_for(task, i)
        ax.plot(
            raw_episodes,
            rewards,
            color=color,
            alpha=0.15,
            linewidth=0.8,
        )

        if mode == "average":
            episodes, summary = sliding_average(rewards, window)
            ylabel = f"Average Reward (window={window})"
            title = "Sliding-Window Training Reward"
        elif mode == "upper":
            episodes, summary = sliding_envelope(rewards, window, mode="upper")
            ylabel = f"Upper Envelope Reward (window={window})"
            title = "Sliding-Window Upper Envelope"
        elif mode == "lower":
            episodes, summary = sliding_envelope(rewards, window, mode="lower")
            ylabel = f"Lower Envelope Reward (window={window})"
            title = "Sliding-Window Lower Envelope"
        else:
            raise ValueError(f"Unsupported plot mode: {mode}")

        ax.plot(
            episodes,
            summary,
            color=color,
            linewidth=2,
            label=f"{task} (n={len(rewards)})",
        )
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No non-empty logs were available to plot")

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_dir = os.path.dirname(os.path.abspath(output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"Saved -> {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory containing JSON log files. Default: logs",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task names to include, without .json. Default: all JSON logs in --log-dir",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Sliding average window size. Default: {DEFAULT_WINDOW}",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("evaluation", "sliding_window_average.png"),
        help="Output image path. Default: evaluation/sliding_window_average.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.window < 1:
        print("--window must be >= 1", file=sys.stderr)
        sys.exit(2)

    logs = find_logs(args.log_dir, args.tasks)
    if not logs:
        print(f"No JSON logs found in {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    outputs = output_paths(args.output)
    plot_reward_summary(logs, args.window, outputs["average"], mode="average")
    plot_reward_summary(logs, args.window, outputs["upper"], mode="upper")
    plot_reward_summary(logs, args.window, outputs["lower"], mode="lower")


if __name__ == "__main__":
    main()
