from env import HOenv
import gymnasium as gym

env = gym.make("HOenv")

obs, info = env.reset(seed=42)


print(env.type)