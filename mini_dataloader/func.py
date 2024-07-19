import cv2
import numpy as np
from .flip import horizontal_flip, vertical_flip
from .rotate import rotate
from .gaussian_blur import gaussian_blur
from .gaussian_noise import gaussian_noise
from .jpeg_compression import jpeg_compression
from .resize import resize

def read(forgery_path, mask_path):
    forgery = cv2.imread(forgery_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return forgery, mask

def process(forgery, mask, augment_prob=0.5, size=(512, 512)):
    """

    :param forgery: Forgery image-
    :param mask:    Binary image. White area represents tampering.
    :param augment: Data augmentation, default is True. Type:{Flip, Rotate}
    :return:
    """

    # _/_/_/ Data Augmentation _/_/_/
    enable_aug_types = [0, 1, 2]  # The kinds of data enhancements allowed
    # The type of data enhancement mixed in a single data
    num_aug_types = np.random.choice(len(enable_aug_types), 1) + 1
    aug_types = np.random.choice(enable_aug_types, num_aug_types, replace=False)
    for aug_type in aug_types:
        if aug_type == 0:
            forgery, mask = horizontal_flip(forgery, mask, p=augment_prob)
        elif aug_type == 1:
            forgery, mask = vertical_flip(forgery, mask, p=augment_prob)
        elif aug_type == 2:
            forgery, mask = rotate(forgery, mask, p=augment_prob)
        elif aug_type == 3:
            forgery = resize(forgery, rates=[0.2], p=augment_prob)
        elif aug_type == 4:
            forgery = jpeg_compression(forgery, qfs=[30], p=augment_prob)
        elif aug_type == 5:
            forgery = gaussian_noise(forgery, sds=[15], p=augment_prob)
        elif aug_type == 6:
            forgery = gaussian_blur(forgery, ksizes=[13], p=augment_prob)

    # _/_/_/ Normalization _/_/_/
    forgery, mask = cv2.resize(forgery, size, cv2.INTER_AREA), cv2.resize(mask, size, cv2.INTER_NEAREST)
    mask = np.where(mask > 127, 255, 0)
    forgery, mask = forgery.astype(np.float32) / 255., mask.astype(np.float32) / 255.

    # _/_/_/ To fit the model inputs _/_/_/
    forgery = forgery[:, :, ::-1]
    mask = np.expand_dims(mask, axis=-1)
    forgery, mask = np.transpose(forgery, (2, 0, 1)), np.transpose(mask, (2, 0, 1))

    return forgery, mask
