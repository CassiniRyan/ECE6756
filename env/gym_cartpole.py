import gymnasium as gym

def make_env(render=False):
    if render:
        return gym.make("CartPole-v1", render_mode="human")
    return gym.make("CartPole-v1")