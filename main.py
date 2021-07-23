import warnings

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from model import Model
from utils import Utils

warnings.filterwarnings('ignore')

utils = Utils()

# Conjunto de datos conformado por diferentes tipo de ropa y artículos de uso.
fashion_mnist = keras.datasets.fashion_mnist
# Separamos las imágenes y etiquetas de entrenamiento y de prueba.
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# Etiquetas (columnas) que trae el DS.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# utils.one_fashion(train_images, n=200)

# Dividimos los datos de entrenamiento y de prueba entre 255 para normalizar la
# escala de color entre 0 y 1 porque la escala de colores va desde 0 hasta 255.
train_images = train_images / 255.0
test_images = test_images / 255.0

# utils.print_fashion(train_images, train_labels, class_names)

modelo = Model(train_images, train_labels,
               test_images, test_labels, class_names)
modelo.acc()
modelo.eval()

# Miremos la imagen [0], sus predicciones y el arreglo de predicciones. Las
# etiquetas de predicción correctas estan en azul y las incorrectas estan en
# rojo. El número entrega el porcentaje (sobre 100) para la etiqueta predecida.
# Graficamos múltiples imágenes con sus predicciones. Notese que el modelo
# puede estar equivocado aún cuando tiene mucha confianza.
img1 = 0
modelo.plot_image(img1)
img2 = 68
modelo.plot_image(img2)

# Muestra varias prendas, su percentaje de exactitud y la predicción escogida.
num_rows = 5
num_cols = 5
modelo.print_all(num_rows, num_cols)
