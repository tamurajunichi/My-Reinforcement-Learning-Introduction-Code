import numpy as np


class ValueIteration:
    def __init__(self, gamma=1.0, delta=0, theta=0.02, v_shape=100):
        self.gamma = gamma
        self.delta = delta
        self.theta = theta
        self.V = np.zeros(v_shape)
        self.v = np.copy(self.V)
        self.policy = self.init_pi()
        self.actions = [i for i in range(100)]
        self.sweep_count = 0

    def init_pi(self):
        pi = []
        for s in range(100):
            if s != 0:
                pi.append([1]*s)
        return pi

    def _action(self, s, index):
        next_s_prob = []
        p = 0.4
        for i in range(2):
            if i == 0:
                next_s_prob.append([s+self.actions[index], p])
            else:
                next_s_prob.append([s-self.actions[index], 1-p])
        return next_s_prob

    def reward(self, next_s):
        if next_s == 99:
            return 1
        else:
            return 0

    def iteration(self):
        while True:
            self.sweep_count += 1
            self.delta = 0
            self.v = np.copy(self.V)
            for s in range(99):# 0 ~ 98
                a = self.policy[s]
                maxindex = [i for i, x in enumerate(a) if x == max(a)]
                array = []
                for index in maxindex:# 0 ~ 98
                    V_s = 0
                    for next_s, prob in  self._action(s+1, index+1):
                        if next_s > 99:
                            next_s = 99
                        V_s += prob * (self.reward(next_s) + self.gamma * self.v[next_s])
                    array.append(V_s)
                self.V[s] = max(array)
            self.delta = np.max(np.absolute(self.v - self.V))
            if self.delta < self.theta:
                break

    def optimize(self):
        """
        ここはまだ未完成、うまく最適方策が出力されていない。
        :return:
        """
        for s in range(99):# 0 ~ 98
            a = self.policy[s]
            maxindex = [i for i, x in enumerate(a) if x == max(a)]
            array = []
            for index in maxindex:# 0 ~ 98
                V_s = 0
                for next_s, prob in  self._action(s+1, index+1):
                    if next_s > 99:
                        next_s = 99
                    V_s += prob * (self.reward(next_s) + self.gamma * self.V[next_s])
                array.append(V_s)
            for _s in range(99):
                self.policy[_s] = [0] * (_s+1)
            max_a_index = array.index(max(array))
            self.policy[s][max_a_index] = 1


def main():
    coin = ValueIteration()
    coin.iteration()
    print(coin.sweep_count)
    print(coin.V)
    coin.optimize()
    print(coin.policy)


if __name__ == "__main__":
    main()
