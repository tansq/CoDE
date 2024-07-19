import cv2
import numpy as np

def rotate(forgery, mask, p=0.5):
    if np.random.rand() <= p:
        i = np.random.randint(0, 3)
        if i == 0:
            forgery = cv2.rotate(forgery, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        elif i == 1:
            forgery = cv2.rotate(forgery, cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            forgery = cv2.rotate(forgery, cv2.ROTATE_180)
            mask = cv2.rotate(mask, cv2.ROTATE_180)

    return forgery, mask
