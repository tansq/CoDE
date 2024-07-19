import numpy as np

def gaussian_noise(forgery, mean=0, sds=[3, 5, 7, 9, 11, 13, 15], p=0.5):
    if np.random.rand() <= p:
        sd = np.random.choice(sds)
        size = forgery.shape
        noise = np.random.normal(mean, sd, size)
        forgery = forgery + noise
        forgery = np.clip(forgery, 0, 255)
    return forgery
