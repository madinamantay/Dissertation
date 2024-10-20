import numpy as np
from keras import utils
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import csv


class Classifier:
    def __init__(self):
        self.x = None
        self.y = None

        self.x_test = None
        self.y_test = None

        self.model = None

        self.history = None

    def set_train_data(self, x, y):
        self.x = x
        self.y = y

    def set_test_data(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def set_model(self, model):
        self.model = model

    def load_model(self, model_path: str):
        self.model.load_weights(model_path)

    def train(self, output_model: str, classes=43, batch_size=32, epochs=150):
        x_train, x_val, y_train, y_val = train_test_split(
            self.x, self.y,
            test_size=0.3, random_state=42, shuffle=True,
        )

        y_train = utils.to_categorical(y_train, classes)
        y_val = utils.to_categorical(y_val, classes)

        aug = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.15,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode="nearest",
        )

        self.history = self.model.fit(
            aug.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_val, y_val),
        )

        self.model.save(output_model)

    def import_metrics(self, file_name: str):
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'acc', 'val_acc', 'loss', 'val_loss'])
            for i in range(len(self.history.history['accuracy'])):
                writer.writerow(
                    [i+1,
                     self.history.history['accuracy'][i],
                     self.history.history['val_accuracy'][i],
                     self.history.history['loss'][i],
                     self.history.history['val_loss'][i],
                     ])

    def test(self):
        predict_x = self.model.predict(self.x_test)
        pred = np.argmax(predict_x, axis=1)

        return accuracy_score(self.y_test, pred) * 100

    def test_once(self, image_path: str):
        data = []
        image = Image.open(image_path).resize((32, 32))
        data.append(np.array(image))
        x = np.array(data)

        predict = self.model.predict(x)

        y = np.argmax(predict, axis=1)
        return y.item()
