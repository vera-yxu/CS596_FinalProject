#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 09:19:45 2018

@author: billy
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
X = np.load('dna_cleaner.npy')
y = np.load('dna_clean_labels.npy')
np.where(y==2)
X = X[:1535]
y = y[:1535]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#X_train.shape
num_inputs = X.shape[0]
input_size = X.shape[1]

def initialize_parameters(input_size):
    # Weights can all start at zero for logistic regression
    weights = np.zeros(input_size,)
    # For any kind of network, we can initialize the bias to zero
    bias = np.zeros(1)
    return (weights, bias)

weights, bias = initialize_parameters(input_size)

#print("The weight parameter is of shape: " + str(weights.shape))
#print("The bias parameter is of shape: " + str(bias.shape))

"""
Forward propagartion

Now that we’ve initialized our variables, it’s time to make an initial calculation. We’re going to take each input value and multiply it by a weight. Then we’ll sum the weights together and add a bias. The bias term can make one outcome more likely than another, which helps when there is more of one outcome than another (such as cancer vs. not cancer in diagnostic imagery).

Here is how it works mathematically:

Z
=
w
0
x
0
+
w
1
x
1
+
.
.
.
w
m
x
m
Z=w 
0
​	 x 
0
​	 +w 
1
​	 x 
1
​	 +...w 
m
​	 x 
m
​	 
where

Z
Z is the sum of the weighted inputs
w
0
w 
0
​	  is the initial weight, often known as the bias
x
0
x 
0
​	  is 1 to allow for the bias 
w
0
w 
0
​	 
w
1
w 
1
​	  is the weight associated with the first input
x
1
x 
1
​	  is the first input
w
m
w 
m
​	  is the weight associated with the last input
x
m
x 
m
​	  is the last input
Thus we can map any value of x to a value for y between 0 and 1. We can then ask if the result is greater than 0.5. If it is, put it in category 1, if not, category 0. And we’ve got our prediction. The tricky part, however, is to find the correct weights.

Let’s make sure there is a weight for every element in the image. The shapes of the ndarrays must be the same.
"""
assert X_train[0].shape == weights.T.shape
image_sum = np.dot(weights, X_train[0]) + bias
#image_sum

def sigmoid(z): #creating the S shaped curve.
    s = 1/(1+np.exp(-z))
    return s

x_sigmoid = np.arange(-5, 5, .1)
y_sigmoid = sigmoid(x_sigmoid)
plt.plot(x_sigmoid,y_sigmoid)
plt.title('Sigmoid Function')
plt.show()