# TODO decide parameters used for this program
import gym
import numpy as np
env = gym.make('MountainCar-v0')


def discretize(_observation,division_num = 100):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / division_num

    position = int((_observation[0] - env_low[0])/env_dx[0])
    velocity = int((_observation[1] - env_low[1])/env_dx[1])
    return position, velocity

class MountainCar:
    def __init__(self,_env):
        self._env = _env
        self.action = 1

    def play(self, _action):
        self._env.render()
        observation, reward, done, info = self._env.step(_action)
        return observation, reward


class Agent:
    def __init__(self, epsilon):
        self.epsilon = epsilon

def main():
    for i in range(20):
        observation = env.reset()
        for t in range(200):
            env.render()
            position,velocity = discretize(observation)
            print("{} : {}".format(position,velocity))
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        env.close()

if __name__ == "__main__":
    main()

