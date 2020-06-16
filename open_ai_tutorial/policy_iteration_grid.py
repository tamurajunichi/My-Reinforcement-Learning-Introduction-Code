import numpy as np
import matplotlib.pyplot as plt


# TODO matplotで描画する。
class PolicyIteration:
    def __init__(self):
        self.gamma = 1.0
        self.delta = 0
        self.theta = 0.5
        self.V = np.zeros(16)
        self.v = np.copy(self.V)
        self.policy = np.ones((16, 4))
        self.old = np.copy(self.policy)
        self.actions = [4, -4, 1, -1]

    def evaluation(self):
        while True:
            self.delta = 0
            self.v = np.copy(self.V)
            for s in range(16):
                V_s = 0
                if not (s == 0 or s == 15):
                    a = self.policy[s]
                    maxindex = [i for i, x in enumerate(a) if x == max(a)]
                    for index in maxindex:
                        next_s = self._action(s, int(self.actions[index]))
                        V_s += self.prob(s) * (self.reward() + (self.gamma * self.v[next_s]))
                    self.V[s] = V_s
            self.delta = np.max(np.absolute(self.v - self.V))
            print(self.delta)
            if self.delta < self.theta:
                break

    def prob(self, s):
        return 0.25

    def reward(self):
        return -1

    def _action(self, s, a):
        _s = s + a
        if a == 4:
            if 11 < s and s < 15:
                _s = _s - a
        elif a == -4:
            if 0 < s and s < 4:
                _s = _s - a
        elif a == 1:
            if s == 3 or s == 7 or s == 11:
                _s = _s - a
        elif a == -1:
            if s == 4 or s == 8 or s == 12:
                _s = _s - a
        return _s

    def improvement(self):
        self.policy_stable = True
        self.old = np.copy(self.policy)
        for s in range(16):
            if not (s == 0 or s == 15):
                array = [0]*4
                for i, a in enumerate(self.actions):
                    next_s = self._action(s, int(a))
                    array[i] = self.prob(s) * (self.reward() + (self.gamma * self.V[next_s]))
                maxindex = [i for i, x in enumerate(array) if x == max(array)]
                for i in range(4):
                    self.policy[s][i] = 0
                for index in maxindex:
                    self.policy[s][index] = 1

        if np.sum(self.old == self.policy) != self.policy.size:
            self.policy_stable = False

        if self.policy_stable:
            return False
        else:
            return True



def main():
    done = True
    grid_world = PolicyIteration()
    while done:
        grid_world.evaluation()
        print(grid_world.V.reshape(4, 4))
        done = grid_world.improvement()
        print(grid_world.policy)


if __name__ == "__main__":
    main()

