# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from keras.models import Sequential
from keras.datasets import fashion_mnist

class_names = [
    "T-shirt/top", 
    "Trouser", 
    "Pullover", 
    "Dress", 
    "Coat", 
    "Sandal", 
    "Shirt", 
    "Sneaker", 
    "Bag", 
    "Ankle boot"
]


class FashionImageClassifier():
    def __init__(self, dataset = fashion_mnist):
        self.dataset = dataset

    # load dataset
    def load_keras_data(self):
        """
        method to load dataset from keras.datasets
        parameters: None
        ----------
        returns: X_train_full, y_train_full, X_test, y_test
        -------
        Example: X_train, y_train, X_test, y_test = load_keras_data()
        -------
        """
        
        # Load the dataset from keras
        (X_train_full, y_train_full), (X_test, y_test) = self.dataset.load_data()
    
        # Return the train-test splits
        return X_train_full, y_train_full, X_test, y_test
        
    
    def preprocess(self, X_train, y_train, split = 0.2, scale_value = 255):
        """
        method to preprocess the loaded dataset. By spliting the train data into validation data.
        Then, scaling the feature matrix: X_train/255, X_val / 255
        
        parameters:
        ----------
            -X_train,
            -y_train,
            -split: The validation size, default: 0.2
            -scale_value: the value for scaling the feature matrix, default: 255
            
        returns: X_train, y_train, X_valid, y_valid
        -------
        Example: X_train, y_train, X_valid, y_valid = preprocess(X_train, y_train, split = 0.1)
        -------
        """
        
        # Adjust Split
        split = round(split * len(X_train))
        
        # Split into Validation sets
        X_valid, X_train = X_train[:split], X_train[split:]
        y_valid, y_train = y_train[:split], y_train[split:]
        
        # Scale between 0 and 1
        X_valid = X_valid / scale_value
        X_train = X_train / scale_value
    
        return X_train, y_train, X_valid, y_valid
    
    def build_model(self):
        """
        method to build the Neural Network. all parameters are defined. 
        The model contains 2 Hidden layers.
        1st layer has 300 neurons, activation = 'relu'
        2nd layer has 100 neurons, activation = 'relu'
        
        parameters: None
        ----------
        returns: model
        -------
        Example: model = build_model()
        -------
        """
        
        # Instantiate Model
        model = Sequential()
        
        # Add Layers to the model
        model.add(keras.layers.Flatten(input_shape = [28, 28]))
        model.add(keras.layers.Dense(300, activation = "relu"))
        model.add(keras.layers.Dense(100, activation = "relu"))
        model.add(keras.layers.Dense(10, activation= "softmax"))
    
        return model
    
    def compile_model(self, model):
        """
        method to compile the Neural Network. all parameters are defined. 
        The model contains 2 Hidden layers.
               
        parameters: model
        ----------
        returns: None
        -------
        Example: compile_model(model)
        -------
        """
        
        # Compiling the model
        model.compile(
            loss = "sparse_categorical_crossentropy",
            optimizer = "sgd",
            metrics = ["accuracy"]
        )
        return "model compiled successfully"
        
    
    
    def train_model(self, X_train, y_train, X_valid, y_valid, model):
        """
        method to train the model by fitting it to the training and validation sets
        
        parameters:
        ----------
            -X_train,
            -y_train,
            -X_valid,
            -y_valid,
            -model
            
        returns: history
        -------
        Example: history = train_model(X_train, y_train, X_valid, y_valid, model)
        -------
        """
        
        # Fitting the Model to the train and Validation datasets
        history = model.fit(
            X_train,
            y_train,
            epochs= 30,
            validation_data= (X_valid, y_valid)
        )
        return history
    
    def save_model(self, model, name = "model.h5"):
        """
        method to save the trained model using keras.model.save
               
        parameters: provide the following:
        ----------
            - model,
            - name: file name to save as, default = "model.h5
            
        returns: None
        -------
        Example: save_model(model, name = "model.h5")
        -------
        """
        
        # Save the Model
        model.save(name)
        return "Model saved successfully"
    
    def load_up_model(self, name):
        """
        method to load_up the saved model using keras.models.load
               
        parameters: provide the:
        ----------
            - name: model name
            
        returns: model
        -------
        Example: new_model = load_up_model('model.h5')
        -------
        """
        
        # load saved Model
        model = keras.models.load_model(name)
    
        return model
    
    def plot_learning_curve(self, model_history):
        """
        method to plot the learning curve of the trained model. this only works directly with the model history.
               
        parameters: provide the:
        ----------
            - name: model_history
            
        returns: graph for learning curve
        -------
        Example: plot_learning_curve(history.history)
        -------
        """
        
        # Learning Curve 
        pd.DataFrame(model_history.history).plot(
            title = "Learning Curve",
            xlabel = "epochs", 
            ylabel = "Accuracy and Loss"
        )
        
    
    def evaluate_prediction(self, model, X_test, y_test):
        """
        method to evaluate the model performance on test data.
               
        parameters: provide these:
        ----------
            - model
            - X_test
            - y_test
            
        returns: loss, accuracy
        -------
        Example: loss, accuracy = evaluate_prediction(model, X_test, y_test)
        -------
        """
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, accuracy
    
    def visualize_some_prediction(self, model, X_test):
        """
        method to visualize figures for three predictions
               
        parameters: provide these:
        ----------
            - model
            - X_test
            
        returns: three figures or images
        -------
        Example: visualize_some_prediction(model, X_test)
        -------
        """
        
        # Make Prediction in One-Hot Encoding
        y_pred = model.predict(X_test[:3])
        
        # Convert Prediction to class labels
        y_pred_label = np.argmax(y_pred, axis = 1)
        
        # Obtaining class name for predicted labels
        class_ = np.array(class_names)[y_pred_label]
        
        # Show three Samples
        plt.figure(figsize=(10, 3))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(X_test[i].reshape(28, 28), cmap = "gray")
            plt.title(f"pred: {np.array(class_names)[y_pred_label[i]]}")
            plt.axis("off")
        
        plt.show()
    
    def display_confusion_matrix(self, model, X_test, y_test):
        """
        method to visualize the confusion matrix display of the model
               
        parameters: provide these:
        ----------
            - model
            - X_test
            - y_test
            
        returns: confusion matrix display
        -------
        Example: display_confusion_matrix(model, X_test, y_test):
        -------
        """
        
        # Prediction
        prediction = np.argmax(model.predict(X_test), axis= 1)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, prediction)
        
        # Confusion Matrix Display
        disp = ConfusionMatrixDisplay(confusion_matrix= cm)
        disp.plot(cmap = "Blues")
        plt.title("Confusion Matrix")
        plt.show()