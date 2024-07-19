import cv2
import numpy as np

def horizontal_flip(forgery, mask, p=0.5):
    if np.random.rand() <= p:
        forgery = cv2.flip(forgery, 1)
        mask = cv2.flip(mask, 1)
    return forgery, mask

def vertical_flip(forgery, mask, p=0.5):
    if np.random.rand() <= p:
        forgery = cv2.flip(forgery, 0)
        mask = cv2.flip(mask, 0)
    return forgery, mask
