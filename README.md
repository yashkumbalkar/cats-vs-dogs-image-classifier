# Cats vs Dogs Classifier using Convolutional Neural Networks (CNN)

This project is a binary image classifier that distinguishes between images of cats and dogs using a Convolutional Neural Network (CNN). The model is trained on a dataset 
of labeled cat and dog images and classifies new images as either a cat or a dog.

## **Dataset**

The dataset used for training and testing the model is the [Kaggle Cats vs. Dogs dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats?select=train).

## **Model Architecture**

The model is a Convolutional Neural Network (CNN) that uses several convolutional layers followed by pooling layers to extract features from the input images.
The CNN architecture consists of:

- Convolutional layers with ReLU activation.
- Max-pooling layers for downsampling.
- Fully connected layers for classification.
- Sigmoid activation at the output layer for binary classification.

## **Tools Used**
- `TensorFlow` (for model building and training).
- `Keras` (for high-level neural network API).
- `NumPy` (for numerical operations).
- `Matplotlib` (for visualization).
- `OpenCV` (for image preprocessing).

## **Result**

The model achieves an accuracy of approximately 82.86% on the test set.
