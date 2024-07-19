import cv2
import numpy as np

def resize(forgery, rates=[0.78, 0.50, 0.25], p=0.5):
    if np.random.rand() <= p:
        if rates is None:
            rate = np.random.choice(np.arange(25, 400, 1)) / 100.
        else:
            rate = np.random.choice(rates)
        ori_size = forgery.shape[:2]
        forgery = cv2.resize(forgery, (int(ori_size[1] * rate), int(ori_size[0] * rate)), cv2.INTER_AREA)
    return forgery
