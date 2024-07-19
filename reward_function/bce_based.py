import numpy as np

# BCE-based Reward
def bce(y_true, y_pred):
    eps = 1e-8
    bce_loss = y_true * (np.log(y_pred + eps)) + (1 - y_true) * (np.log(1 - y_pred + eps))
    return - bce_loss
def bce_reward(previous_mask, current_mask, gt):
    reward = bce(gt, previous_mask) - bce(gt, current_mask)
    return reward