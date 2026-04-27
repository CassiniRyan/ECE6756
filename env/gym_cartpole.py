# Environment factory for CartPole using Gymnasium.
#
# Keeping environment construction in one place avoids small differences between
# algorithms. Every training run uses the same CartPole-v1 setup.
import gymnasium as gym

def make_env(render=False):
    """Create a CartPole environment, optionally with human rendering."""
    if render:
        return gym.make("CartPole-v1", render_mode="human")
    return gym.make("CartPole-v1")
