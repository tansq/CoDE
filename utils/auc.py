from sklearn.metrics import roc_auc_score
import numpy as np

def cal_auc(outputs, gt):
    batch_size = outputs.shape[0]
    auc = 0
    for idx in range(batch_size):
        trans_output = outputs[idx].flatten()
        trans_gt = gt[idx].flatten().astype(np.int32)
        try:
            roc_auc = roc_auc_score(trans_gt, trans_output)
        except ValueError:
            roc_auc = 0
        auc += roc_auc
    auc = auc / batch_size

    return auc


if __name__ == '__main__':
    outputs = np.array([[0, 1, 0, 1, 0, 1, 1, 0, 1, 0]], dtype=np.int32)
    gt = np.array([[0, 0, 1, 1, 0, 1, 1, 1, 1, 0]], dtype=np.int32)
    outputs = outputs.reshape((1, 2, 5))
    gt = gt.reshape((1, 2, 5))
    print(outputs.shape, gt.shape)
    print(cal_auc(outputs, gt))
