from keras import layers, models, Input
from tensorflow.keras.optimizers import Adam


def get_res_net_18(num_classes=43, shape=(32, 32, 3), lr=0.001):
    def identity_block(input_tensor, kernel_size, filters):
        filters1, filters2 = filters
        x = layers.Conv2D(filters1, kernel_size, padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
        filters1, filters2 = filters
        x = layers.Conv2D(filters1, kernel_size, strides=strides, padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)

        shortcut = layers.Conv2D(filters2, (1, 1), strides=strides, padding='same')(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    inputs = Input(shape=shape, name="input_1")

    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    x = identity_block(x, 3, [64, 64]) # (64, 64) filters for the two conv layers in block
    x = identity_block(x, 3, [64, 64])

    x = conv_block(x, 3, [128, 128], strides=(2, 2)) # Downsamples
    x = identity_block(x, 3, [128, 128])

    x = conv_block(x, 3, [256, 256], strides=(2, 2)) # Downsamples
    x = identity_block(x, 3, [256, 256])

    x = conv_block(x, 3, [512, 512], strides=(2, 2)) # Downsamples
    x = identity_block(x, 3, [512, 512])

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='fc1000')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet18_Scratch')

    opt = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
