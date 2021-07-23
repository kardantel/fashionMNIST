import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')


class Utils:
    '''
    Class that contains auxiliar methods used by the classifier.
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
        # 10x10 images are plotted to avoid displaying them in real size.
        plt.figure(figsize=(10, 10))
        # 25 images are plotted.
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid('off')
            # Training images. Grayscale color.
            plt.imshow(train_images[i], cmap='Greys')
            # Tag or group to which the displayed image belongs is displayed.
            plt.xlabel(class_names[train_labels[i]])
        plt.show()
