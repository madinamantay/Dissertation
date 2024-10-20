import tensorflow as tf

from data import load_class


def load_dataset(path: str, classes=43, shape=(32, 32, 3)) -> tf.data.Dataset:
    data = []
    for label in range(classes):
        images = load_class(path, label)
        for image in images:
            data.append((image, label))

    return tf.data.Dataset.from_generator(lambda: data,
                                          output_types=(tf.float32, tf.int32),
                                          output_shapes=(shape, ()))
