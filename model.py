import warnings

from tensorflow import keras
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Model:
    '''
    Export
    '''

    def __init__(self, train_images, train_labels, test_images, test_labels,
                 class_names):
        '''
        Returns
        '''
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.class_names = class_names

        # Definimos el modelo con el que vamos a entrenar.
        # Cada una de estas secuencias es una capa de la red neuronal.

        # "Aplanamos" cada una de las imágenes de 28x28.
        # Creamos una primera capa: aplicamos una densidad a las capas de 128 y
        # una FA ReLU.
        # Creamos una segunda capa: se encarga de aprender un poco más con
        # densidad 10 y FA softmax.
        self.model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                                       keras.layers.Dense(
                                           128, activation=tf.nn.relu),
                                       keras.layers.Dense(10,
                                                          activation=tf.nn.softmax)])

        # Creamos el compilador del modelo con el optimizador 'Adam' (una
        # variación avanzada de Batch Gradient Descent).
        # Optimizador que también puede ser llamado como 'Adam()' si antes se
        # importa la biblioteca correspondiente. Esto es como el modelo se
        # actualiza basado en el set de datos que ve y la funcion de perdida.
        # Mide qué tan exacto es el modelo durante el entrenamiento. La idea es
        # minimizar el error para dirigir el modelo en la direccion adecuada.
        # Se usa para monitorear los pasos de entrenamiento y de pruebas. Se
        # usa accuracy (exactitud), la fraccion de la imagenes que son
        # correctamente clasificadas.
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        # Entrenamos el modelo con las imágenes y etiquetas de entrenamiento.
        # 'epochs' es la cantidad de iteraciones que se hacen para entrenar el
        # modelo.
        self.model.fit(self.train_images, self.train_labels, epochs=5)

    def acc(self):
        '''
        The model is evaluated with the test images and labels.
        '''
        self.test_loss, self.test_acc = self.model.evaluate(self.test_images,
                                                            self.test_labels)
        print('Accuracy:', self.test_acc)

    def eval(self, print_img=False):
        '''
        'predictions[i]' nos retorna un array con las probabilidades de
        pertenecer a cada uno de los labels, es decir, la probabilidad de que
        sea cada prenda (bastaría multiplicar por 100 para sacar el porcentaje)
        por lo que la predicción viene a ser la probabilidad máxima en ese
        array; y lo obtenemos con 'np.argmax' que nos dará el índice del valor
        máximo del array, entonces ya tendríamos el índice de la predicción y
        bastaría con obtener el nombre de la prenda usando ese índice en la
        lista de los nombres: 'class_names[predicted_label]'.
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
                # Si la etiqueta de prueba es igual a la predicha, el color
                # será azul; de lo contrario será roja.
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
