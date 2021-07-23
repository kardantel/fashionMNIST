import warnings

from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np

warnings.filterwarnings('ignore')


class Utils:
    '''
    Class that contains all the methods used by the classifiers.
    '''
    @classmethod
    def one_fashion(cls, train_images, num=0):
        '''
        Allows to separate the dataframe based on an initial and a final index.

        ## Parameters
        train_images: array-like of shape (n_samples, n_features)
            Train images from fashion MNIST. Can be for example a list,
            or an array.
        num: int, default=`0`
        '''
        plt.figure()
        plt.title(f'Wearable inside MNIST. Element {num}')
        plt.imshow(train_images[num])
        plt.grid()
        plt.show()

    @classmethod
    def print_fashion(cls, train_images, train_labels, class_names):
        '''
        Print 25 different wearable inside MNIST.
        '''
        # Ploteamos imágenes 10x10 para evitar que las muestre en tamaño real.
        plt.figure(figsize=(10, 10))
        # Ploteamos 25 imágenes.
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid('off')
            # Imágenes de entrenamiento. Color en blanco y negro.
            plt.imshow(train_images[i], cmap='Greys')
            # Muestra la etiqueta o grupo al que pertenece la imagen mostrada.
            plt.xlabel(class_names[train_labels[i]])
        plt.show()
