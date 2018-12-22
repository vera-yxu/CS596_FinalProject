
"""
Created on Wed Dec 19 09:19:45 2018

@author: Will
"""
#libraries
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#open first half
X = np.load('dna_cleaner.npy')
y = np.load('dna_clean_labels.npy')
np.where(y==2)
X = X[:1535]
y = y[:1535]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
num_inputs = X.shape[0]
input_size = X.shape[1]

def initialize_parameters(input_size):
    weights = np.zeros(input_size,)
    bias = np.zeros(1)
    return (weights, bias)

weights, bias = initialize_parameters(input_size)


""" Forward propagartion """

def sigmoid(z): #creating the S shaped curve.
    s = 1/(1+np.exp(-z))
    return s

x_sigmoid = np.arange(-5, 5, .1)
y_sigmoid = sigmoid(x_sigmoid)

#plt.plot(x_sigmoid,y_sigmoid)
#plt.title('Sigmoid Function')
#plt.show()
#print(output)
#y_prediction = np.rint(output).astype(int)
#print(y_prediction)
#loss = -(y_train[0] * np.log(output) + (1-y_train[0])*np.log(1-output))
learning_rate = 0.001
#weights[0] = weights[0] + learning_rate * loss * X_train[0][0]
weights, bias = initialize_parameters(input_size)
outputs = sigmoid(np.dot(weights,X_train.T) + bias)
y_prediction = np.rint(outputs).astype(int) # this forces either 0 or 1


""" Backward Propagation """

cost = (-1/num_inputs) * np.sum(y_train*np.log(outputs) + (1-y_train)*np.log(1-outputs))
dw = (1/num_inputs) * np.dot(X_train.T, (outputs-y_train).T)
db = (1/num_inputs) * np.sum(outputs-y_train)
assert weights.shape == dw.shape
weights = weights - learning_rate * dw
bias = bias - learning_rate * db


""" Building model """

def model(X_train, y_train, epochs=500, learning_rate=0.1):

    num_inputs = len(X_train)

    # Create an empty array to store predictions
    y_prediction = np.empty(num_inputs)

    weights, bias = initialize_parameters(input_size)

    for i in range(epochs):

        # Now we calculate the output
        # We're doing this for all images at the same time
        image_sums = np.dot(weights, X_train.T) + bias
        # Now we have to run each output through our activation function, then convert it to a prediction
        outputs = sigmoid(image_sums)
        # Now we have to convert the outputs to predictions
        # round it to an int
        y_prediction = np.rint(outputs).astype(int)  # this forces either 0 or 1

        # Find weight and bias changes
        dw = (1/num_inputs) * np.dot(X_train.T, (outputs-y_train).T)
        dw = np.squeeze(dw)
        db = (1/num_inputs) * np.sum(outputs-y_train)
        # Update the parameters
        # Make sure the matrices are the same size
        assert weights.shape == dw.shape
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

    parameters = {"weights": weights,
                  "bias": bias,
                  "train_predictions": y_prediction,
                  }

    return parameters

"""Test"""

results = model(X_train, y_train, epochs=300)
final_weights = results['weights']
final_bias = results['bias']
final_sums = np.dot(final_weights, X_test.T) + final_bias
final_outputs = sigmoid(final_sums)
test_predictions = np.rint(final_outputs).astype(int) 
train_predictions = results['train_predictions']
print("Accuracy on training set: {:.2%}".format(1-np.mean(np.abs(train_predictions - y_train))))
print("Accuracy on testing set: {:.2%}".format(1-np.mean(np.abs(test_predictions - y_test))))

