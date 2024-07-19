import cv2
import numpy as np

def jpeg_compression(forgery, qfs=[60, 70, 80, 90, 95], p=0.5):
    if np.random.rand() <= p:
        if qfs is None:
            qf = np.random.choice(np.arange(70, 100, 1))
        else:
            qf = np.random.choice(qfs)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qf]
        result, encimg = cv2.imencode('.jpg', forgery, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg
    return forgery