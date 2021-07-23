import warnings

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from model import Model
from utils import Utils

warnings.filterwarnings('ignore')

utils = Utils()


def load_mnist(print_img=False):
    # Conjunto de datos conformado por diferentes tipo de ropa y
    # artículos de uso.
    fashion_mnist = keras.datasets.fashion_mnist
    # Separamos las imágenes y etiquetas de entrenamiento y de prueba.
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    # Separamos las imágenes y etiquetas de entrenamiento y de prueba.
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    # Etiquetas (columnas) que trae el DS.
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Dividimos los datos de entrenamiento y de prueba entre 255 para
    # normalizar la escala de color entre 0 y 1 porque la escala de colores va
    # desde 0 hasta 255.
    train_images, test_images = train_images / 255.0, test_images / 255.0

    if print_img:
        utils.one_fashion(train_images, num=200)
        utils.print_fashion(train_images, train_labels, class_names)

    return train_images, test_images, train_labels, test_labels, class_names


def plot_example(modelo):
    '''
    The correct prediction labels are in blue and the incorrect prediction
    labels are in red.
    The number returns the percentage (out of 100) for the predicted label.
    Multiple images are graphed with their predictions.
    Note that the model can be wrong even when it is very confident.
    '''
    # Correctamente clasificada.
    img1 = 1
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    modelo.plot_image(img1)
    plt.subplot(2, 2, 2)
    modelo.plot_value_array(img1)
    # Incorrectamente clasificada.
    img2 = 68
    plt.subplot(2, 2, 3)
    modelo.plot_image(img2)
    plt.subplot(2, 2, 4)
    modelo.plot_value_array(img2)
    plt.show()


def main():
    # Se Cargan los datos de MNIST
    train_images, test_images, train_labels, test_labels, class_names = load_mnist()
    # Se instancia el modelo
    modelo = Model(train_images, train_labels,
                   test_images, test_labels, class_names)
    # Se calcula la exactitud y se evalúa el modelo con 25 imágenes del MNIST.
    modelo.acc()
    modelo.eval()   # Si se quieren ver las imágenes hacer 'print_img=True'
    # Se grafican dos imágenes de prueba: una correctamente clasificada y otra no
    plot_example(modelo)
    # Muestra varias prendas, su percentaje de exactitud y la predicción escogida.
    num_rows = 5
    num_cols = 4
    modelo.print_all(num_rows, num_cols)


if __name__ == "__main__":
    main()
