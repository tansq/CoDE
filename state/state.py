import numpy as np

class State:
    def __init__(self):
        self.state = None
        self.mask = None

    def reset(self, forgery):
        b, c, h, w = forgery.shape
        self.state = np.zeros((b, c+1, h, w), dtype=np.float32)
        self.mask = np.zeros((b, 1, h, w), dtype=np.float32)
        self.state[:, :-1, :, :] = forgery
        self.state[:, -1:, :, :] = self.mask

    def step(self, act):
        actions = act.astype(np.float32)
        self.mask = np.clip(self.mask + actions, a_min=0, a_max=1.0)
        self.state[:, -1:, :, :] = self.mask

