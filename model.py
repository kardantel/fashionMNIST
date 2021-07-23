import warnings

from tensorflow import keras
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Model:
    def __init__(self, train_images, train_labels, test_images, test_labels,
                 class_names):
        '''
        Contains the Keras model for classification, compiler and model
        testing.
        '''
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.class_names = class_names

        # The model with which we are going to train is defined.
        self.model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                                       keras.layers.Dense(
                                           128, activation=tf.nn.relu),
                                       keras.layers.Dense(10,
                                                          activation=tf.nn.softmax)])

        # We create the model compiler with the 'Adam' optimizer (an advanced
        # variation of Batch Gradient Descent).
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        # We train the model with the training images and labels.
        self.model.fit(self.train_images, self.train_labels, epochs=5)

    def acc(self):
        '''
        Accuracy calculation is performed with test images and labels.
        '''
        self.test_loss, self.test_acc = self.model.evaluate(self.test_images,
                                                            self.test_labels)
        print('Accuracy:', self.test_acc)

    def eval(self, print_img=False):
        '''
        The prediction is made with the test images.
        You can choose to print 25 images used for the prediction with the
        `print_img` parameter.

        ## Parameters
        print_img: boolean [optional], defualt=False
            If `True` it prints 25 test images used in the prediction.
        '''
        self.predictions = self.model.predict(self.test_images)

        if print_img:
            plt.figure(figsize=(10, 10))
            for i in range(25):
                plt.subplot(5, 5, i + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid('off')
                plt.imshow(self.test_images[i], cmap='Greys')
                predicted_label = np.argmax(self.predictions[i])
                true_label = self.test_labels[i]
                # If the test label is the same as predicted,the color will be
                # blue; otherwise it will be red.
                if predicted_label == true_label:
                    color = 'blue'
                else:
                    color = 'red'

                plt.xlabel('{} ({})'.format(self.class_names[predicted_label],
                                            self.class_names[true_label]),
                           color=color)
            plt.show()

    def plot_image(self, i):
        '''
        Prints the garment, its real name, the hit percentage and the
        classified label.
        If the label was correctly classified, the text is printed in blue,
        otherwise in red.
        '''
        true_label, img = self.test_labels[i], self.test_images[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap='Greys')

        predicted_label = np.argmax(self.predictions[i])
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
                                             100 * np.max(self.predictions[i]),
                                             self.class_names[true_label]),
                   color=color)

    def plot_value_array(self, i):
        '''
        Prints the graph that indicates which label it was classified on.
        If the label was correctly classified, the text is printed in blue,
        otherwise in red.
        '''
        true_label = self.test_labels[i]
        plt.grid(False)
        plt.xticks(range(10), self.class_names, rotation=45)
        plt.yticks([])
        thisplot = plt.bar(range(10), self.predictions[i], color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(self.predictions[i])

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    def print_all(self, num_rows, num_cols):
        '''
        Plots the first X test images, their predicted labels, and the
        true labels.
        Color the correct predictions in blue and the incorrect ones in red.
        '''
        # num_rows = 9
        # num_cols = 7
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plot_image(i)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plot_value_array(i)
        plt.tight_layout()
        plt.show()
