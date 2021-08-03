# Wearable classification from Fashion MNIST dataset using Keras neural network
This repository contains the code made to create a wearable image classifier, obtained from the Keras Fashion MNIST dataset. The classification is done using a neural network created in Keras, with a hidden layer and 10 neurons in the output layer. The classifier achieves accuracy values greater than 85%.

### Features

- Use Keras from Tensorflow in version 2.5.0.
- Shows the category to which each of the wearables that the user wants to know belongs.
- Allows you to observe wearables that have been correctly classified and those that have not.
- Correctly classified wearables are shown in blue; those that don't, in red.
- Of the garments that have been incorrectly classified, it shows the category in which it was classified and the true category.
- The code allows the user to create an image with a size of rows and columns to their liking with classified garments.

## How to use
In `main.py` you can find the 4 main methods:
- `load_mnist(print_img=True)` allows you to load the Fashion MNIST dataset with necessary pretreatment. Print 2 images: one is garment 200 from the dataset.
![](https://i.imgur.com/k5GjZcl.png)
The other image prints a sample of 25 grayscale images.
![](https://i.imgur.com/fPZ4mko.png)
- `plot_example()` prints the result of sorting into two wearable items: one correctly sorted and the other not.
![](https://i.imgur.com/eYy45h8.png)
- `acc()` make the Accuracy calculation is performed with test images and labels. In this case, an accuracy of **87,6%** was achieved.
- `print_all(num_rows, num_cols)` prints the classification obtained for a certain number of usable, depending on the number of rows and columns. In this case, values `num_rows=5` and` num_cols=4` were chosen to obtain an image with 20 classified images.
![](https://i.imgur.com/abpUmW2.png)
