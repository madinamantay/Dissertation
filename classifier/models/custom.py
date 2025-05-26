from keras import layers, models
from tensorflow.keras.optimizers import Adam


def get_classifier_model(num_classes=43, shape=(32, 32, 3), lr=0.001):
    model = models.Sequential([
        layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=shape),
        layers.Conv2D(32, kernel_size=3, activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.BatchNormalization(axis=-1),

        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.Conv2D(128, kernel_size=3, activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.BatchNormalization(axis=-1),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.5),

        layers.Dense(num_classes, activation='softmax')
    ])

    opt = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
