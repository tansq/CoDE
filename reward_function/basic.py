def basic_reward(previous_mask, current_mask, gt):
    coefficient = 1
    gt_ = gt * coefficient
    gt_[gt_ <= 0] = -1
    reward = (current_mask - previous_mask) * gt_
    return reward