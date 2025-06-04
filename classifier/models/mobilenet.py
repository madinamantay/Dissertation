import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam


def get_mobilenet_v2(num_classes=43, input_shape=(32, 32, 3), lr=0.001, alpha=1.0, weights='imagenet'):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        pooling='avg'
    )

    if weights == 'imagenet':
        base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name=f'mobilenetv2_gtsrb_alpha_{alpha}')

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
