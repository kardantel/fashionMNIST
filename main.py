"""
Created by kardantel at 8/8/2020
__author__ = 'Carlos Pimentel'
__email__ = 'carlosdpimenteld@gmail.com'
"""

import warnings

from tensorflow import keras

import matplotlib.pyplot as plt

from model import Model
from utils import Utils

warnings.filterwarnings('ignore')

utils = Utils()


def load_mnist(print_img=False):
    '''
    Loads the dataset made up of different types of clothing and wearable from
    the MNIST dataset.

    ## Parameters
    print_img: boolean [optional], defualt=False
        Print a sample image and 25 different wearable inside MNIST.
    '''
    fashion_mnist = keras.datasets.fashion_mnist
    # Training and test images and labels are separated.
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    # Labels (columns) that the DS brings.
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # The training and test data are divided by 255 to normalize the color
    # scale between 0 and 1 because the color scale ranges from 0 to 255.
    train_images, test_images = train_images / 255.0, test_images / 255.0

    if print_img:
        utils.one_fashion(train_images, num=200)
        utils.print_fashion(train_images, train_labels, class_names)

    return train_images, test_images, train_labels, test_labels, class_names


def plot_example(model):
    '''
    The correct prediction labels are in blue and the incorrect prediction
    labels are in red.
    The number returns the percentage (out of 100) for the predicted label.
    Multiple images are graphed with their predictions.
    Note that the model can be wrong even when it is very confident.

    ## Parameters
    model: object
        Model that will be used to plot de images calling the functions.
    '''
    # Correctly classified.
    img1 = 1
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    model.plot_image(img1)
    plt.subplot(2, 2, 2)
    model.plot_value_array(img1)
    # Incorrectly classified.
    img2 = 68
    plt.subplot(2, 2, 3)
    model.plot_image(img2)
    plt.subplot(2, 2, 4)
    model.plot_value_array(img2)
    plt.show()


def main():
    '''
    Main function.
    '''
    # MNIST data is loaded
    train_images, test_images, train_labels, test_labels, class_names = load_mnist()
    # The model is instantiated
    model = Model(train_images, train_labels,
                  test_images, test_labels, class_names)
    # The accuracy is calculated and the model is evaluated with 25 images from
    # the MNIST.
    model.acc()
    model.eval()   # If you want to see the images, do 'print_img=True'
    # Two test images are graphed: one correctly classified and the other not
    plot_example(model)
    # Various garments are shown, their percentage of accuracy and the chosen
    # prediction.
    num_rows = 5
    num_cols = 4
    model.print_all(num_rows, num_cols)


if __name__ == "__main__":
    main()
