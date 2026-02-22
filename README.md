# CIFAR-10 Image Classification with CNN
This project demonstrates the development and evaluation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset.

Table of Contents
Introduction
Dataset
Model Architecture
Training
Results
Setup and Usage
Introduction
This notebook implements a CNN to classify images from the CIFAR-10 dataset into one of 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The goal was to achieve at least 75% accuracy on the test set.

Dataset
The project uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

Data Preprocessing
Images are loaded using datasets.cifar10.load_data().
Pixel values are normalized by dividing by 255.0 to scale them between 0 and 1.
Model Architecture
The final CNN architecture used in this project is defined as follows:

model = models.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dropout(0.5),

    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])
The model uses Conv2D layers with ReLU activation, followed by MaxPooling2D layers for downsampling.
Dropout layers are incorporated after pooling and before the dense layers to reduce overfitting.
The final layers consist of Flatten to convert the 3D output to 1D, and Dense layers, with the last one having 10 units for the 10 classes.
Training
The model was compiled with:

Optimizer: adam
Loss Function: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
Metrics: accuracy
The model was trained for 30 epochs.

Results
After training for 30 epochs, the model achieved a test accuracy of:

Test Accuracy: 0.7547 (75.47%)

This result meets the project requirement of achieving at least 75% accuracy on the CIFAR-10 dataset.

Setup and Usage
To run this notebook:

Clone the repository or download the .ipynb file.
Open in Google Colab or a Jupyter environment.
Ensure necessary libraries are installed:
pip install tensorflow matplotlib numpy
Run all cells in the notebook sequentially. The notebook will download the CIFAR-10 dataset, define and train the model, and evaluate its performance.
