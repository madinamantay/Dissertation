from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam


def identity_block(x, filters):
    f1, f2 = filters

    shortcut = x

    x = layers.Conv2D(f1, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(f2, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def conv_block(x, filters, strides=(2, 2)):
    f1, f2 = filters

    shortcut = layers.Conv2D(f2, (1, 1), strides=strides, padding='same')(x)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(f1, (3, 3), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(f2, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def get_res_net_18(num_classes=43, input_shape=(32, 32, 3), lr=0.001):
    inputs = Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = conv_block(x, [64, 64], strides=(1, 1))
    x = identity_block(x, [64, 64])

    x = conv_block(x, [128, 128])
    x = identity_block(x, [128, 128])

    x = conv_block(x, [256, 256])
    x = identity_block(x, [256, 256])

    x = conv_block(x, [512, 512])
    x = identity_block(x, [512, 512])

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet18_GTSRB')

    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
