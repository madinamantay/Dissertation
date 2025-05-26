import cv2
import numpy as np
import pandas as pd

import os
import shutil

def load_class(path: str, class_label: int):
    train_path = os.path.join(path, 'Train')
    train_images = load_train(train_path, class_label)
    test_images = load_test(path, class_label)

    train_images = np.concatenate([train_images, test_images])
    train_images = train_images.astype('float32')
    train_images = (train_images - 127.5) / 127.5

    return train_images


def load_train(path: str, class_label: int, dim: tuple = (32, 32)):
    images = []
    class_path = os.path.join(path, str(class_label))
    for img_name in os.listdir(class_path):
        if img_name is not None:
            im = cv2.imread(os.path.join(class_path, img_name))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, dim)
            images.append(im)

    return np.asarray(images)


def load_test(path: str, class_label: int, dim: tuple = (32, 32)):
    images = []
    test_csv = pd.read_csv(os.path.join(path, 'Test.csv'))
    filtered_test_csv = test_csv[test_csv['ClassId'] == class_label]
    filenames = filtered_test_csv['Path'].tolist()

    for file in filenames:
        im = cv2.imread(os.path.join(path, file))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, dim)
        images.append(im)

    return np.asarray(images)

def copy_images_from_two_directories(dir1, dir2, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)

    shutil.copytree(dir1, destination, dirs_exist_ok=True)
    shutil.copytree(dir2, destination, dirs_exist_ok=True)
