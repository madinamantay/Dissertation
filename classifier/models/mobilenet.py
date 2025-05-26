from keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2


def get_mobilenet_v2(num_classes=43, shape=(32, 32, 3), lr=0.001):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=shape)
    base_model.trainable = False

    model = models.Sequential(name="MobileNetV2_Pretrained")
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D()) # Reduce feature maps to a single vector per channel
    model.add(layers.Dense(num_classes, activation='softmax'))

    opt = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
