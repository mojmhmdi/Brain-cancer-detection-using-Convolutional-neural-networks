
import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from PIL import Image


def read_and_resize_images(path, size, label):
    images = []
    labels = []
    dir_list = (os.listdir(path))
    for i in range(len(dir_list)):
        img = Image.open(path + '/' + str(dir_list[i]))
        img = img.resize((size, size))
        img = np.array(img)
        if (img.shape == (size, size, 3)):
            images.append(img)
            labels.append(one_hot_label(label))
    return images, labels


def one_hot_label(labels):
    encoder = OneHotEncoder()
    encoder.fit([[0], [1]])
    return encoder.transform([[labels]]).toarray()
