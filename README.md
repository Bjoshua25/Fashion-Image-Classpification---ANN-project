# 👚🧠 Fashion Image Classifier using ANN | Fashion MNIST

This project implements a **Fashion MNIST image classifier** using a **simple Artificial Neural Network (ANN)** built with TensorFlow and Keras. The model learns to classify grayscale images of clothing items into 10 categories, such as shirts, sneakers, and coats.

---

## 📁 Project Structure

- `fashion_MNIST_project.ipynb` – Jupyter Notebook containing the full workflow: data loading, preprocessing, model building, training, and evaluation.

---

## 📦 Dataset

The project uses the **Fashion MNIST** dataset:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- 10 categories of clothing

Provided by: [`tf.keras.datasets.fashion_mnist`](https://www.tensorflow.org/datasets/catalog/fashion_mnist)

---

## 🏗️ Model Architecture

A simple **feedforward neural network (ANN)** built with Keras:
- Input Layer: Flatten (28×28 → 784)
- Hidden Layer: Dense with ReLU activation
- Output Layer: Dense with 10 neurons (softmax activation)

> 🔧 Optimizer: sdg  
> 📉 Loss: Sparse Categorical Crossentropy  
> 🎯 Metrics: Accuracy

---

## 📊 Performance

- Achieved test accuracy: **~84%**
- Suitable for basic classification tasks with relatively low computational cost.

---

## 📈 Results Visualization

The notebook includes:
- Training and validation accuracy/loss plots
- Sample image predictions with predicted vs true labels
- Confusion matrix to evaluate per-class performance
