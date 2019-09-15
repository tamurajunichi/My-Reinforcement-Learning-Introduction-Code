import numpy as np
import random


class ValueIteration:
    def __init__(self, gamma, theta, v_shape, pi_shape):
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(v_shape)
        self.v = np.copy(self.V)
        self.delta = 0
        self.pi_s = np.ones(pi_shape)

    def sweep(self):
        self.v = self.V
        for s in range(1, self.V.size):
            self.V[s-1] = self.max_a_state_value(s)
        self.delta = np.max(np.absolute(self.v - self.V))

    def max_a_state_value(self, s):
        actions = np.arange(1, s+1, 1)
        array = np.zeros(s)
        for i, action in enumerate(actions):
            array[i] = self.state_value(s, action)
        return np.max(array)

    def state_value(self, s, action):
        V_s = 0
        for p in [0.4, 0.6]:
            if p == 0.4:
                _s = s + action
            else:
                _s = s - action

            if _s > 99 or _s < 1:
                V_s += p * (self.reward(_s) + (self.gamma * 0))
            else:
                V_s += p * (self.reward(_s) + (self.gamma * self.V[_s]))
        return V_s

    def reward(self, s):
        if s > 99:
            return 1
        else:
            return 0

    def iteration(self):
        while True:
            self.delta = 0
            self.sweep()
            if self.delta < self.theta:
                break
        for s in range(1, self.pi_s.size):
            actions = np.arange(1, s+1, 1)
            array = np.zeros(s)
            for i, action in enumerate(actions):
                array[i] = self.state_value(s, action)
            self.pi_s[s-1] = actions[np.argmax(array)]


def main():
    coin = ValueIteration(gamma=0.1, theta=1, v_shape=100)
    coin.iteration()
    print(coin.pi_s)


if __name__ == "__main__":
    main()
