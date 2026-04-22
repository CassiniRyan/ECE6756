
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
        choices=["q", "dyna-q", "dyna-q-discret", "dyna-q-linear", "rwm-q"],
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
        model = LinearModel(action_space=env.action_space.n)
        planning_steps = 6

    elif args.algo == "rwm-q":
        from models.rwm_model import RWMModel
        from third_party.world_model_torch import WorldModel

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Neural world model used by the reactive world-model planner.
        wm = WorldModel(state_dim=4, action_dim=1, num_actions=env.action_space.n, device=device)
        model = RWMModel(wm)
        planning_steps = 1

    agent = Agent(env, model=model, planning_steps=planning_steps, bins_per_dim=bins_per_dim)

    if args.mode == "train":
        # Train the agent for the requested number of episodes.
        if args.mode == "train":
            rewards = agent.train(args.episodes)

            import json
            import os

##################################################
            import matplotlib.pyplot as plt

            # Save a quick visual summary after each training run.
            plt.plot(rewards)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"Training Curve ({algo_name})")
            plt.grid()

            plt.savefig(f"{algo_name}_debug.png")  # safe on server
            print(f"Saved plot -> {algo_name}_debug.png")

            agent.save("q_table.npy")
####################################################

            os.makedirs("logs", exist_ok=True)

            log_data = {
                "algo": algo_name,
                "episodes": args.episodes,
                "rewards": rewards
            }

            # Persist logs and weights so later plots do not need retraining.
            with open(f"logs/{algo_name}.json", "w") as f:
                json.dump(log_data, f)

            agent.save(f"logs/q_table_{algo_name}.npy")

        agent.save("q_table.npy")

    elif args.mode == "run":
        # Load a saved Q-table and execute the policy.
        agent.load("q_table.npy")
        agent.run()

if __name__ == "__main__":
    main()
