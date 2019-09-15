import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def poisson(lamda, n):
    return (lamda**n)*math.exp(-lamda)/math.factorial(n)


class PolicyIteration:
    def __init__(self, gamma, theta, v_shape, actions):
        self.gamma = gamma
        self.theta = theta
        self.delta = 0
        self.V = np.zeros(v_shape)
        self.v = None
        self.actions = actions
        self.pi_s = np.zeros(v_shape)
        self.policy_stable = True
        self.done = True

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
                    row, col = int(row), int(col)
                    self.V[row][col] = self.state_value(row, col)
        self.delta = np.max(np.absolute(self.v - self.V))

    def state_value(self, row, col, action=None):
        V_s = 0
        if action is None:
            action = self.pi_s[row][col]

        next_row_start = row - action
        next_col_start = col + action
        if not (next_col_start < 0 or next_col_start > 20 or next_row_start < 0 or next_row_start > 20):
            next_row_start, next_col_start = int(next_row_start), int(next_col_start)
            for row_rental in range(next_row_start):
                for col_rental in range(int(next_col_start)):
                    row_rest = next_row_start - row_rental
                    col_rest = next_col_start - col_rental
                    for row_return in range(20 - row_rest):
                        for col_return in range(20 - col_rest):

                            trans_prob = self.prob(row_rental, col_rental, row_return, col_return)
                            V_s += trans_prob * (self.reward(row_rental, col_rental, action) + (self.gamma * self.V[row][col]))
        return V_s


    def prob(self, row_rental, col_rental, row_return, col_return):
        p = poisson(3, row_rental)*poisson(3, col_rental)*poisson(4, row_return)*poisson(2, col_return)
        return p

    def reward(self, row_rental, col_rental, action):
        r = (row_rental + col_rental) * 10 - action * 2
        return r

    def iteration(self):
        old = np.copy(self.pi_s)
        array = np.zeros(len(self.actions))
        self.policy_stable = True

        for row in range(self.pi_s.shape[0]):
            for col in range(self.pi_s.shape[1]):
                if (row != 0 and col != 0) or (row != 3 and col != 3):

                    old[row][col] = self.pi_s[row][col]
                    for i, action in enumerate(self.actions):
                        array[i] = self.state_value(row, col, action=action)
                    self.pi_s[row][col] = self.actions[array.argmax()]

                    if old[row][col] != self.pi_s[row][col]:
                        self.policy_stable = False

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
    jack = PolicyIteration(gamma=0.9, theta=1, v_shape=(21, 21), actions=list(range(-5, 6)))
    np.set_printoptions(precision=1, suppress=True)
    while True:
        print("----evaluation----")
        jack.evaluation()
        print(jack.V)
        print()
        print("----iteration----")
        done = jack.iteration()
        print(jack.pi_s)
        print()
        jack.plot()
        if done is True:
            break
    print("finish")



if __name__ == "__main__":
    main()
