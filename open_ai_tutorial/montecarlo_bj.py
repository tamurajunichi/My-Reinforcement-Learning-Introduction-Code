import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


DEALER_CARD = 11
PLAYER_CARD = 10
USABLE_ACE = 2

def get_card():
    card = random.randint(1, 13)
    if card > 10:
        card = 10
    return card

class BlackJack:
    def __init__(self, agent):
        self.done = False
        self.step = 0
        self.agent = agent
        self.final_episode = 500000
        self.V = np.zeros((2, 10, 10))# player12～21, dealer1～10
        self.R = np.zeros((2, 10, 10))
        self.counter = np.zeros((2, 10, 10))# sへの訪問回数

    def play(self):
        episode = 0
        while episode < self.final_episode:
            episode += 1
            player_card, dealer_card, log_states = self.agent.play()
            reward = self._reward(player_card, dealer_card)
            p_state = []
            d_state = []
            usable_ace = []
            for i, log_state in enumerate(log_states):
                if log_state[0] < 22:
                    p_state.append(log_state[0] -12)
                    d_state.append(log_state[1] - 1)
                    usable_ace.append(log_state[2])
                    self.R[usable_ace[i]][p_state[i]][d_state[i]] += reward
                    self.counter[usable_ace[i]][p_state[i]][d_state[i]] += 1
            for p,d,u in zip(p_state, d_state, usable_ace):
                self.V[u][p][d] = self.R[u][p][d] / self.counter[u][p][d]

        print(self.R)
        print(self.R[1][8][9])
        print(self.V)

    def _reward(self,player_card,dealer_card):
        p_sums = sum(player_card)
        d_sums = sum(dealer_card)

        if p_sums > 21:
            return -1
        if d_sums > 21:
            return 1
        if p_sums > d_sums:
            return 1
        elif p_sums == d_sums:
            return 0
        else:
            return -1

    def plot(self,i):
        x = np.arange(0,10)
        y = np.arange(0,10)
        X, Y = np.meshgrid(x, y)
        Z = self.V[i]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("Dealer")
        ax.set_ylabel("Player")
        ax.set_zlabel("State Value")
        ax.plot_surface(X, Y, Z)
        plt.show()


class Agent:
    def __init__(self):
        self.dealer_card = []
        self.player_card = []
        self.state = []
        self.dealer_policy = 17
        self.player_policy = 20
        self.episode = 0
        self.step = 0
        self.usable_ace = 0

    def play(self):
        usable_ace = 0
        self.log_state = []
        self.dealer_card = self.init_play_dealer()
        self.player_card = self.init_play_player()

        # player turn
        while True:
            #1が出たか確認
            usable_ace = 0
            if 1 in self.player_card and sum(self.player_card) < 12:
                usable_ace = 1
            if sum(self.player_card) >= self.player_policy:
                if self.log_state == []:
                    self.log_state.append([sum(self.player_card), self.dealer_card[0], usable_ace])
                #print("PLAYER :{}".format(sum(self.player_card)))
                break
            self.play_player(usable_ace)

        # dealer turn
        while True:
            #1が出たか確認
            usable_ace = 0
            if 1 in self.dealer_card and sum(self.dealer_card) < 12:
                usable_ace = 1
            if sum(self.dealer_card) >= self.dealer_policy:
                #print("DEALER :{}".format(sum(self.dealer_card)))
                break
            self.play_dealer(usable_ace)

        return [self.player_card, self.dealer_card, self.log_state]

    def init_play_dealer(self):
        card = []
        for i in range(2):
            card.append(get_card())
        return card

    def init_play_player(self):
        card = []
        for i in range(2):
            card.append(get_card())
        return card

    def play_player(self, usable_ace):
        # 2枚でブラックジャックの場合
        sum_card = sum(self.player_card)
        if usable_ace == 1 and (sum_card == 11 or sum_card == 10):
            self.player_card[self.player_card.index(1)] = 11 # 1を11に変換
            #print("player_card: {}".format(sum(self.player_card)))
            self.log_state.append([sum(self.player_card), self.dealer_card[0], usable_ace])
        # ブラックジャックしてないときで2枚
        # くず手にならないで11にできるなら常に11にしておく
        elif usable_ace == 1 and sum_card < 10:
            # 1～8の時にしか変換しないので12～19になる　この範囲でしか11にできない
            # Aが2枚来たときよう　片方のみ11にする
            indexs = [i for i, x in enumerate(self.player_card) if x == 1]
            self.player_card[indexs[0]] = 11
            #print("player_card: {}".format(sum(self.player_card)))
            self.log_state.append([sum(self.player_card), self.dealer_card[0], usable_ace])
            self.player_card.append(get_card())
            sum_card = sum(self.player_card)
            #print("player_card: {}".format(sum(self.player_card)))
            self.log_state.append([sum(self.player_card), self.dealer_card[0], usable_ace])
        # 1がないとき
        elif usable_ace == 0:
            self.player_card.append(get_card())
            sum_card = sum(self.player_card)
            #print("player_card: {}".format(sum(self.player_card)))
            self.log_state.append([sum(self.player_card), self.dealer_card[0], usable_ace])


    def play_dealer(self, usable_ace):
        # 2枚でブラックジャックの場合
        sum_card = sum(self.dealer_card)
        if usable_ace == 1 and (7 <= sum_card and sum_card <= 11):
            self.dealer_card[self.dealer_card.index(1)] = 11 # 1を11に変換
            #print("dealer_card: {}".format(sum(self.dealer_card)))
        # ブラックジャックしてないときで2枚
        # くず手にならないで11にできるなら常に11にしておく
        elif usable_ace == 1 and sum_card < 7:
            indexs = [i for i, x in enumerate(self.dealer_card) if x == 1]
            self.dealer_card[indexs[0]] = 11
            #print("dealer_card: {}".format(sum(self.dealer_card)))
            self.dealer_card.append(get_card())
            sum_card = sum(self.dealer_card)
            #print("dealer_card: {}".format(sum(self.dealer_card)))
        # 1がないとき
        elif usable_ace == 0:
            self.dealer_card.append(get_card())
            sum_card = sum(self.dealer_card)
            #print("dealer_card: {}".format(sum(self.dealer_card)))

def main():
    agent = Agent()
    bj = BlackJack(agent)
    bj.play()
    for i in range(2):
        bj.plot(i)


if __name__ == "__main__":
    main()
