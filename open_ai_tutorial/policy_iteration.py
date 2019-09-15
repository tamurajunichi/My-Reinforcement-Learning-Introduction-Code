import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def poisson(lamda, n):
    return (lamda**n)*math.exp(-lamda)/math.factorial(n)


class PolicyIteration:
    def __init__(self, gamma, theta, v_shape):
        self.gamma = gamma
        self.theta = theta
        self.delta = 0
        self.V = np.zeros(v_shape)
        self.v = None
        self.actions = {"up":0, "down":1, "left":2, "right":3}#
        self.pi_s = np.zeros((v_shape[0],v_shape[1],int(len(self.actions))))
        self.policy_stable = True
        self.done = True

    def _action(self, _s_row, _s_col, action):
        if action is "up":
            _s_row = _s_row - 1
            if _s_row < 0:
                _s_row = _s_row + 1
        elif action is "down":
            _s_row = _s_row + 1
            if _s_row > 3:
                _s_row = _s_row - 1
        elif action is "left":
            _s_col = _s_col - 1
            if _s_col < 0:
                _s_col = _s_col + 1
        elif action is "right":
            _s_col = _s_col + 1
            if _s_col > 3:
                _s_col = _s_col - 1
        return _s_row, _s_col



    def evaluation(self):
        while True:
            self.sweep()
            print("delta : "+ str(self.delta))
            if self.delta < self.theta:
                break

    def sweep(self):
        self.v = np.copy(self.V)
        for row in range(self.V.shape[0]):
            for col in range(self.V.shape[1]):
                if (row != 0 and col != 0) or (row != 3 and col != 3):
                    row, col = int(row), int(col)
                    #print("row:{},col:{}".format(row,col))
                    self.V[row][col] = self.state_value(row, col)
        self.delta = np.max(np.absolute(self.v - self.V))

    def state_value(self, row, col, action=None):
        V_s = 0
        if action == None:
                if self.pi_s[row][col].size != 1:
                    for action in self.pi_s[row][col]:
                        _s_row, _s_col = row, col
                        _s_row, _s_col = self._action(row, col, list(self.actions.keys())[int(action)])
                        V_s += self.prob() * (self.reward() + (self.gamma * self.V[_s_row][_s_col]))
                else:
                    action = self.pi_s[row][col]
                    _s_row, _s_col = row, col
                    _s_row, _s_col = self._action(row, col, list(self.actions.keys())[int(action)])
                    V_s += self.prob() * (self.reward() + (self.gamma * self.V[_s_row][_s_col]))
        else:
            _s_row, _s_col = row, col
            _s_row, _s_col = self._action(row, col, action)
            V_s += self.prob() * (self.reward() + (self.gamma * self.V[_s_row][_s_col]))
        return V_s

    def policy(self, row, col):
        return self.pi_s[row][col]

    def prob(self):
        return 0.25

    def reward(self):
        return -1

    def iteration(self):
        old = np.copy(self.pi_s)
        array = np.zeros(len(self.actions))
        self.policy_stable = True

        for row in range(self.pi_s.shape[0]):
            for col in range(self.pi_s.shape[1]):
                if (row != 0 and col != 0) or (row != 3 and col != 3):

                    old[row][col] = self.pi_s[row][col]
                    for i, action in enumerate(list(self.actions.keys())):
                        array[i] = self.state_value(row, col, action=action)
                    for i, maxindex in enumerate([i for i, x in enumerate(array) if x == max(array)]):
                        self.pi_s[row][col][i] = maxindex

                    #if old[row][col] != self.pi_s[row][col]:
                     #   self.policy_stable = False

        if self.policy_stable:
            return True
        else:
            return False

    def plot(self):
        x = np.arange(0, self.V.shape[0])
        y = np.arange(0, self.V.shape[1], 1)
        X, Y = np.meshgrid(x, y)
        Z = self.V
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("State Value")
        ax.plot_surface(X, Y, Z)
        plt.show()


def main():
    grid = PolicyIteration(gamma=1, theta=1.1, v_shape=(4,4))
    np.set_printoptions(precision=2, suppress=True)
    while True:
        print("----evaluation----")
        grid.evaluation()
        print(grid.V)
        print()
        print("----iteration----")
        done = grid.iteration()
        print(grid.pi_s)
        print()
        grid.plot()
        if done is True:
            break
    print("finish")



if __name__ == "__main__":
    main()
