import tensorflow as tf
from keras import layers


def get_generator(classes=43):
    con_label = layers.Input(shape=(1,))
    label_embedding = layers.Embedding(classes, 100)(con_label)
    nodes = 4 * 4
    label_dense = layers.Dense(nodes)(label_embedding)
    latent_vector_output = layers.Reshape((4, 4, 1))(label_dense)

    latent_vector = layers.Input(shape=(100,))
    nodes = 512 * 4 * 4
    latent_dense = layers.Dense(nodes)(latent_vector)
    latent_dense = layers.ReLU()(latent_dense)
    label_output = layers.Reshape((4, 4, 512))(latent_dense)

    merge = layers.Concatenate()([latent_vector_output, label_output])

    x = layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', use_bias=False)(merge)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02)(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02)(x)
    x = layers.ReLU()(x)

    out_layer = layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False, activation='tanh')(x)

    return tf.keras.Model([latent_vector, con_label], out_layer)


def get_discriminator(shape=(32, 32, 3), classes=43):
    con_label = layers.Input(shape=(1,))
    label_embedding = layers.Embedding(classes, 100)(con_label)
    nodes = shape[0] * shape[1] * shape[2]
    label_dense = layers.Dense(nodes)(label_embedding)
    label_condition_output = layers.Reshape((shape[0], shape[1], shape[2]))(label_dense)

    inp_image_output = layers.Input(shape=shape)

    merge = layers.Concatenate()([inp_image_output, label_condition_output])

    x = layers.Conv2D(256, kernel_size=4, strides=3, padding='same', use_bias=False)(merge)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(512, kernel_size=4, strides=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    flattened_out = layers.Flatten()(x)
    dropout = layers.Dropout(0.4)(flattened_out)
    dense_out = layers.Dense(1, activation='sigmoid')(dropout)

    return tf.keras.Model([inp_image_output, con_label], dense_out)

