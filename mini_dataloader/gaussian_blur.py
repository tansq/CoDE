import cv2
import numpy as np

def gaussian_blur(forgery, ksizes=[3, 5, 7, 9, 11, 13, 15], p=0.5):
    if np.random.rand() <= p:
        k = np.random.choice(ksizes)
        # forgery = cv2.GaussianBlur(forgery, ksize=(k, k), sigmaX=0)
        forgery = cv2.GaussianBlur(forgery, ksize=(k, k), sigmaX=k*1.0 / 6)
    return forgery
