import numpy as np

from data import load_train, load_test


def load_train_dataset(path: str, classes=43):
    image_data = []
    image_labels = []
    for label in range(classes):
        images = load_train(path, label)
        for image in images:
            image_data.append(image)
            image_labels.append(label)

    image_data = np.array(image_data)
    image_labels = np.array(image_labels)

    shuffle_indexes = np.arange(image_data.shape[0])
    np.random.shuffle(shuffle_indexes)
    image_data = image_data[shuffle_indexes]
    image_labels = image_labels[shuffle_indexes]

    return image_data, image_labels


def load_test_dataset(path: str, classes=43):
    image_data = []
    image_labels = []
    for label in range(classes):
        images = load_test(path, label)
        for image in images:
            image_data.append(image)
            image_labels.append(label)

    image_data = np.array(image_data)
    image_labels = np.array(image_labels)

    shuffle_indexes = np.arange(image_data.shape[0])
    np.random.shuffle(shuffle_indexes)
    image_data = image_data[shuffle_indexes]
    image_labels = image_labels[shuffle_indexes]

    return image_data, image_labels
