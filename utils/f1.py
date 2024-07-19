import numpy as np

def f1_score(y_true, y_pred):
    e = 1e-6
    gp = np.sum(y_true)
    tp = np.sum(y_true*y_pred)
    pp = np.sum(y_pred)
    p = tp/(pp+e)
    r = tp/(gp+e)
    f1 = (2 * p * r) / (p + r + e)
    return f1


def cal_f1(outputs, gt):
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    precision = true_positives / (true_positives + false_positives)
    recall    = true_positives / (true_positives + false_negatives)

    :param outputs: Network Output Mask
    :param gt:      Mask GroudTruth
    :return:
    """
    batch_size = outputs.shape[0]
    f1 = 0
    for idx in range(batch_size):
        f1 += f1_score(gt[idx].flatten(), outputs[idx].flatten())
    f1 = f1 / batch_size

    return f1

