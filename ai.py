import sys
import os
import random
import math
import numpy as np
np.random.seed(1)
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from PIL import Image

class AI(object):
    def __init__(self):
        super(AI, self).__init__()

        self.init_default()

    def init_default(self):
        # mnist = tf.keras.datasets.mnist
        # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # x_train, x_test = x_train.reshape(-1, 28, 28, 1) / 255.0, x_test.reshape(-1, 28, 28, 1) / 255.0

        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1024, activation='relu'))
        self.model.add(layers.Dense(10))

        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        # print('Training Model')
        # history = self.model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

        # model.save_weights("model/my_model.ckpt")
        # print('Model Saved')

        load_status = self.model.load_weights("model/my_model.ckpt")
        print('Model Reloaded')

        # loss, acc = self.model.evaluate(x_test, y_test, verbose=2)
        # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    def predict_image_with_path(self, file_path):
        try:
            im = Image.open(file_path).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)  # resize the image
            im = np.array(im)  # convert to an array
            im = np.ones(im.shape) * 255 - im
            im2 = im / np.max(im).astype(float)  # normalise input

            # with self.graph.as_default():
            test_image = np.reshape(im2, [1, 28, 28, 1])  # reshape it to our input placeholder shape
            p_ = self.model.predict(test_image).argmax(axis=1)
            print('predict numer: {}'.format(p_))
            return '是{}!'.format(p_[0])
        except:
            print('predict_image_with_path error')
            return '錯誤...
