# Plotting helper for comparing saved training reward logs.
import json
import matplotlib.pyplot as plt
import numpy as np

def moving_avg(x, window=50):
    """Smooth a reward curve with a moving average."""
    return np.convolve(x, np.ones(window)/window, mode='valid')

def load_rewards(path):
    """Load the reward list from one saved JSON log."""
    with open(path, "r") as f:
        data = json.load(f)
    return data["rewards"]

# Load the training logs we want to compare.
q = load_rewards("q.json")
dyna = load_rewards("dyna-q.json")

# Smooth the raw per-episode reward curves.
q_s = moving_avg(q)
dyna_s = moving_avg(dyna)

# Plot the comparison and save it for headless/server use.
plt.figure()
plt.plot(q_s, label="Q-learning")
plt.plot(dyna_s, label="Dyna-Q")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Comparison")
plt.legend()
plt.grid()

plt.savefig("comparison.png")  # safe for server
print("Saved plot → comparison.png")
