import numpy as np
import matplotlib.pyplot as plt


def init(row,col):
    return np.zeros(shape=(row,col))

def pi_sa():
    return 0.25

def Pr():
    return 1


def reward(row, col):
    return -1

def _action(_s_row, _s_col, action):
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

def state_value1(row,col):
    V_s = 0
    for action in actions:
        _s_row,_s_col = row, col
        _s_row,_s_col = _action(row, col, action)
        R = reward(_s_row, _s_col)
        V_s += pi_sa() * Pr() * (R + V[_s_row][_s_col])
    return V_s

def sweep1():
    global v
    v = np.copy(V)
    for s in range(V.size):
        if 0 < s and s < 15:
            row, col = int(s / 4),int(s % 4)# sから行と列を割り出す
            V[row][col] = state_value1(row,col)# update()
    delta =  np.max(np.absolute(v - V))
    return delta

def my_plt(k,delta):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(V, cmap=plt.cm.jet,
                    interpolation='nearest')
    width, height = V.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(V[x][y],1)), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    cb = fig.colorbar(res)
    plt.xticks(range(width))
    plt.yticks(range(height))
    plt.ylim(-0.5,3.5)
    ax.invert_yaxis()
    plt.title('k={},delta={}'.format(k,delta))
    plt.show()

V = init(4,4)
v  = init(4,4)
gamma = 1
actions = {"up":-1, "down":1, "left":-1, "right":1}# up:-4,down:4,left:-1,right:1

def main():
    theta = 0.0001
    k =0

    while True:
        delta = 0
        delta = sweep1()
        k = k + 1
        if k < 30:
            print('k={},delta={}'.format(k,delta))
            print(V)
            print()
            my_plt(k,delta)
        if delta < theta:
            break
    my_plt(k,delta)

if __name__ == "__main__":
    main()
