import json
import matplotlib.pyplot as plt
import numpy as np

def moving_avg(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def load_rewards(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["rewards"]

# load logs
q = load_rewards("q.json")
dyna = load_rewards("dyna-q.json")

# smooth
q_s = moving_avg(q)
dyna_s = moving_avg(dyna)

# plot
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