import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time

ACTION_HIT = 0
ACTION_STAY = 1
ACTIONS = [ACTION_HIT, ACTION_STAY]


class BlackJack:
    def __init__(self):
        self.Q = np.zeros((10, 10, 2, 2))# Q(s,a)
        self.R = np.copy(self.Q) #累積報酬
        self.Qc = np.copy(self.Q)# Q(s,a)のカウント用
        self.trajectory = []

    def play(self, episodes, agent):
        for episode in tqdm(range(episodes)):
            trajectory = []
            initial_state = [np.random.choice([0, 1]), np.random.choice(range(12, 22)), np.random.choice(range(1, 11))]
            initial_action = np.random.choice(ACTIONS)

            player_sum = initial_state[1]
            dealer_sum = initial_state[2]

            dealer_card =[]
            dealer_card.append(initial_state[2])

            state = initial_state
            state.append(initial_action)

            # TODO stateに21を超えたplayer_sumの値が行ってしまう
            while True:
                action = agent.policy(episode, self.Q, state)

                if initial_action is not None:
                    action = initial_action
                    initial_action = None

                trajectory.append([state[0], state[1], state[2], action])

                if action == ACTION_HIT:
                    player_sum += self.get_card()
                else:
                    break

                state[1], state[3] = player_sum, action

            while True:
                # dealerの2枚目のカード
                dealer_card.append(self.get_card())
                dealer_sum = sum(dealer_card)
                if 1 in dealer_card:
                    if dealer_sum <= 11:
                        dealer_card[dealer_card.index(1)] = 11
                if dealer_sum > 16:
                    break

            r = self.reward(player_sum, dealer_sum)
            self.state_action_value(r, trajectory)

    def state_action_value(self, reward, trajectory):
        for usable_ace, player_sum, dealer_card, action in trajectory:
            player_sum -= 12
            dealer_card -= 1
            if player_sum > 9:
                break
            self.R[player_sum][dealer_card][usable_ace][action] += reward
            self.Qc[player_sum][dealer_card][usable_ace][action] += 1
        self.Q = self.R / self.Qc

    def get_card(self):
        return random.randint(1, 14)

    def reward(self, player_sum, dealer_sum):
        if player_sum > 21:
            return -1
        else:
            if dealer_sum > 21:
                return 1
            else:
                if player_sum < dealer_sum:
                    return -1
                elif player_sum > dealer_sum:
                    return 1
                else:
                    return 0


class Agent:
    def __init__(self):
        pass

    def policy(self, episode, Q, state):
        action = self.greedy_policy(Q, state) if episode else self.initial_policy(state)
        return action

    def greedy_policy(self, Q, state):
        player = state[1] - 12
        if player > 9:
            return ACTION_STAY
        dealer = state[2] - 1
        usable_ace = state[0]
        return np.argmax(Q[player][dealer][usable_ace])

    def initial_policy(self, state):
        player_sum = state[1]
        if player_sum == 21 or player_sum == 20:
            return ACTION_STAY
        elif player_sum > 11 and player_sum < 20:
            return ACTION_HIT
        else:
            return ACTION_STAY


def main():
    policy = np.zeros((2, 10, 10))

    bj = BlackJack()
    agent = Agent()
    bj.play(50000, agent)

    for i in range(2):
        for j in range(10):
            for k in range(10):
                policy[i][j][k] = np.argmax(bj.Q[j][k][i])

    time.sleep(1)

    print(policy[0])
    print(policy[1])

    x = np.array(range(1,11))
    y = np.array(range(12,21))
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.set_xlabel("dealer open card")
    axis.set_ylabel("player sum")
    X, Y = np.meshgrid(x, y)


if __name__ == "__main__":
    main()