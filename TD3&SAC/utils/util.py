import numpy as np


class Replay_buffer():

    def __init__(self, max_size=2048 * 32):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def deal_main_data(get_tcp_state):
    get_state = get_tcp_state
    distance_terminal = get_state[0]
    posture = get_state[1]
    distance = get_state[2]
    warning = get_state[3]
    block = get_state[4]
    overturn = get_state[5]
    reach = get_state[6]
    velocity = get_state[7]
    destination_angle = get_state[8]
    return distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )