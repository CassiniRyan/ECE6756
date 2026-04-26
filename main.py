
# main.py: entry point for the CartPole reinforcement learning experiment.
# It parses CLI arguments, configures the environment, picks the algorithm,
# initializes the agent, and either trains or runs the learned policy.
import argparse
from env.gym_cartpole import make_env
from rl_core.agent import Agent

def parse_args():
    """Parse command line options for mode, algorithm, episodes, and rendering."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "run"], required=True)
    parser.add_argument(
        "--algo",
        choices=["q", "dyna-q", "dyna-q-discret", "dyna-q-linear", "rwm-q", "rwm-predict"],
        default="q",
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    # Create the environment once and pass it to the agent.
    env = make_env(render=args.render)

    # Normalize the alias so "dyna-q" maps to the discrete/tabular variant.
    algo_name = "dyna-q-discret" if args.algo == "dyna-q" else args.algo
    bins_per_dim = 10

    if args.algo == "q":
        model = None
        planning_steps = 0

    elif algo_name == "dyna-q-discret":
        from models.tabular_model import TabularModel
        # Classic Dyna-Q: memorize discrete transitions and replay them for planning.
        model = TabularModel(max_per_state=8)
        planning_steps = 1
        bins_per_dim = 20

    elif algo_name == "dyna-q-linear":
        from models.linear_model import LinearModel
        # Continuous linear model: fit one-step dynamics in observation space,
        # then discretize the imagined transitions before Q-updates.
        bins_per_dim = 12
        model = LinearModel(action_space=env.action_space.n, mse_threshold=0.02)
        planning_steps = 2

    elif args.algo in ("rwm-q", "rwm-predict"):
        from models.rwm_model import RWMModel, RWMPredictModel
        from third_party.world_model_torch import WorldModel

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Neural world model used by the reactive world-model planner.
        wm = WorldModel(state_dim=4, action_dim=1, num_actions=env.action_space.n, device=device)
        model_cls = RWMPredictModel if args.algo == "rwm-predict" else RWMModel
        model = model_cls(wm)
        planning_steps = 1 if args.algo == "rwm-predict" else 3

    agent = Agent(env, model=model, planning_steps=planning_steps, bins_per_dim=bins_per_dim, log_name=algo_name)

    if args.mode == "train":
        # Train the agent for the requested number of episodes.
        if args.mode == "train":
            rewards = agent.train(args.episodes)

            import json
            import os

##################################################
            import matplotlib.pyplot as plt
            import numpy as np
            debug_dir = "debug"
            os.makedirs(debug_dir, exist_ok=True)

            # Save a quick visual summary after each training run.
            plt.plot(rewards)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"Training Curve ({algo_name})")
            plt.grid()

            debug_plot_path = os.path.join(debug_dir, f"{algo_name}_debug.png")
            plt.savefig(debug_plot_path)  # safe on server
            print(f"Saved plot -> {debug_plot_path}")

            agent.save("q_table.npy")

            # --- RWM debug plots: one-step and multi-step real vs predicted values ---
            if algo_name in ("rwm-q", "rwm-predict") and model is not None and hasattr(model, "debug_prediction_snapshot"):
                snapshot = model.debug_prediction_snapshot()
                if snapshot is not None:
                    pred_next = snapshot["pred_next"]
                    real_next = snapshot["real_next"]
                    labels = snapshot["labels"]

                    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                    axes = axes.flatten()

                    for i, label in enumerate(labels):
                        ax = axes[i]
                        ax.scatter(real_next[:, i], pred_next[:, i], s=12, alpha=0.55)
                        lo = min(real_next[:, i].min(), pred_next[:, i].min())
                        hi = max(real_next[:, i].max(), pred_next[:, i].max())
                        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
                        ax.set_title(f"{label}: predicted vs real")
                        ax.set_xlabel("Real next value")
                        ax.set_ylabel("Predicted next value")
                        ax.grid(True, alpha=0.3)

                    fig.tight_layout()
                    prediction_debug_path = os.path.join(debug_dir, f"{algo_name}_one_step_prediction_debug.png")
                    fig.savefig(prediction_debug_path)
                    plt.close(fig)
                    print(f"Saved plot -> {prediction_debug_path}")

                if hasattr(model, "multistep_prediction_snapshot"):
                    multi_snapshot = model.multistep_prediction_snapshot()
                    if multi_snapshot is not None:
                        labels = multi_snapshot["labels"]
                        pred_by_step = multi_snapshot["pred_by_step"]
                        real_by_step = multi_snapshot["real_by_step"]
                        mae_by_step = multi_snapshot["mae_by_step"]
                        bin_acc_by_step = multi_snapshot["bin_acc_by_step"]
                        rollout_horizon = len(pred_by_step)

                        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                        axes = axes.flatten()
                        final_pred = pred_by_step[-1]
                        final_real = real_by_step[-1]

                        for i, label in enumerate(labels):
                            ax = axes[i]
                            ax.scatter(final_real[:, i], final_pred[:, i], s=12, alpha=0.55)
                            lo = min(final_real[:, i].min(), final_pred[:, i].min())
                            hi = max(final_real[:, i].max(), final_pred[:, i].max())
                            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
                            ax.set_title(f"{rollout_horizon}-step {label}: predicted vs real")
                            ax.set_xlabel("Real future value")
                            ax.set_ylabel("Predicted future value")
                            ax.grid(True, alpha=0.3)

                        steps = list(range(1, rollout_horizon + 1))
                        axes[4].plot(steps, mae_by_step, marker="o")
                        axes[4].set_title("Multi-step rollout MAE")
                        axes[4].set_xlabel("Prediction step")
                        axes[4].set_ylabel("Mean absolute error")
                        axes[4].grid(True, alpha=0.3)

                        acc_values = [
                            np.nan if value is None else value
                            for value in bin_acc_by_step
                        ]
                        axes[5].plot(steps, acc_values, marker="o")
                        axes[5].set_title("Multi-step Q-bin accuracy")
                        axes[5].set_xlabel("Prediction step")
                        axes[5].set_ylabel("Bin accuracy")
                        axes[5].set_ylim(0.0, 1.05)
                        axes[5].grid(True, alpha=0.3)

                        fig.tight_layout()
                        multi_debug_path = os.path.join(debug_dir, f"{algo_name}_multi_step_prediction_debug.png")
                        fig.savefig(multi_debug_path)
                        plt.close(fig)
                        print(f"Saved plot -> {multi_debug_path}")
####################################################

            agent.save(f"logs/q_table_{algo_name}.npy")

        agent.save("q_table.npy")

    elif args.mode == "run":
        # Load a saved Q-table and execute the policy.
        agent.load("q_table.npy")
        agent.run()

if __name__ == "__main__":
    main()
