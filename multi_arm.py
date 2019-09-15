import numpy as np
import matplotlib.pyplot as plt
import random

game_num = 800
episode = 50

class Slotmachine:

    def __init__(self, arms, finish=30):
        self.arms = arms
        self.finish = finish
        self.state = True
        self.play_counter = 0

    def play(self, arm):

        self.play_counter = self.play_counter + 1

        if self.play_counter >= self.finish:
            self.state = False

        if random.random() < self.arms[arm]:
            reward = 1
        else:
            reward = 0
        return arm, reward


class Agent:
    def __init__(self, env, epsilon):
        self.epsilon = epsilon
        self.env = env
        # reward[arm].appendでアームごとに報酬を格納していける。
        self.reward = [[]for i in range(len(env.arms))]
        self.expected = [0] * len(env.arms)

    def action(self):
        while self.env.state:
            arm, reward = self.env.play(self.epsilon_greedy())
            self.reward[arm].append(reward)

        cumulative_reward = 0
        for r in self.reward:
            cumulative_reward += sum(r)

        self.update()

        return cumulative_reward

    def epsilon_greedy(self):
        if random.random() < self.epsilon:
            arm =  random.randint(0, len(self.env.arms)-1)
        else:
            arm = self.policy()
        return arm

    def policy(self):
        return self.expected.index(max(self.expected))

    def update(self):
        for i in range(len(self.env.arms)):
            self.expected[i] = sum(self.reward[i]) * self.env.arms[i]
        self.reward = [[]for i in range(len(self.env.arms))]


if __name__ == "__main__":
    arms = [0.1, 0.5, 0.8, 0.2, 0.7]
    epsilons = [0.0, 0.1, 0.2, 0.3, 0.8]
    cumulative_rewards = [[]for i in range(len(epsilons))]
    slot1 = Slotmachine(arms, finish=game_num)
    for epsilon in epsilons:
        agent = Agent(env=slot1, epsilon=epsilon)

        for j in range(episode):
            cumulative_reward = agent.action()
            cumulative_rewards[epsilons.index(epsilon)].append(cumulative_reward)
            slot1.play_counter = 0
            slot1.state = True

    n_cumulative_rewards = np.array(cumulative_rewards)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("epsilon-greedy",fontsize=16)
    ax.set_xlabel("episode", size=14, weight="light")
    ax.set_ylabel("rewards", size=14, weight="light")
    x = np.arange(0, episode, 1)
    for i in range(len(epsilons)):
        ax.plot(x, n_cumulative_rewards[i], label='epsilon'+ str(epsilons[i]))
    ax.legend()
    plt.show()