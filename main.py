
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
    parser.add_argument("--algo", choices=["q", "dyna-q", "rwm-q"], default="q")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    # Create the environment once and pass it to the agent.
    env = make_env(render=args.render)

    # Select the algorithm and any planning model.
    if args.algo == "q":
        model = None
        planning_steps = 0

    elif args.algo == "dyna-q":
        from models.tabular_model import TabularModel
        model = TabularModel()
        planning_steps = 1

    elif args.algo == "rwm-q":
        from models.rwm_model import RWMModel
        from third_party.world_model_torch import WorldModel

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        wm = WorldModel(state_dim=4, action_dim=1, device=device)
        model = RWMModel(wm)
        planning_steps = 1

    agent = Agent(env, model=model, planning_steps=planning_steps)

    if args.mode == "train":
        # Train the agent for the requested number of episodes.
        if args.mode == "train":
            rewards = agent.train(args.episodes)

            import json
            import os

##################################################
            import matplotlib.pyplot as plt

            plt.plot(rewards)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"Training Curve ({args.algo})")
            plt.grid()

            plt.savefig(f"{args.algo}_debug.png")  # safe on server
            print(f"Saved plot → {args.algo}_debug.png")

            agent.save("q_table.npy")
####################################################

            os.makedirs("logs", exist_ok=True)

            log_data = {
                "algo": args.algo,
                "episodes": args.episodes,
                "rewards": rewards
            }

            with open(f"logs/{args.algo}.json", "w") as f:
                json.dump(log_data, f)

            agent.save(f"logs/q_table_{args.algo}.npy")

        agent.save("q_table.npy")

    elif args.mode == "run":
        # Load a saved Q-table and execute the policy.
        agent.load("q_table.npy")
        agent.run()

if __name__ == "__main__":
    main()
