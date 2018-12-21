from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


with open('splice.csv', 'r') as infile:
    data = infile.readlines()
    data = [i.replace('\n', ' ').replace(' ', '').split(',') for i in data]
    df = pd.DataFrame(data)
    # Drop name column
    df = df.drop(columns=[1])

    enc_dict = {
        'A': np.int('1'),
        'T': np.int('2'),
        'G': np.int('3'),
        'C': np.int('4'),
        'D': np.int('5'),
        'N': np.int('6'),
        'S': np.int('7'),
        'R': np.int('8')
    }
    new_col = []
    enc_seq = []
    for i in df[2]:
        for x in i:
            enc_seq.append(enc_dict[x])
        new_col.append(enc_seq)
        enc_seq = []
    df[2] = new_col
    # Base pairs stored in single matrix
    df2 = pd.DataFrame(df[2].values.tolist())
    df2['cat'] = df[0]
    # Base pairs split into separate columns
    df2.to_csv("data.csv", encoding='utf-8', index=False)



dataset = pd.read_csv('data.csv')
dataset["cat"]=dataset["cat"].astype('category')
i = pd.DataFrame(dataset)
seq = i.values


X = seq[0:, 0:60].astype(float)
Y = seq[0:, 60]
# Set Seed
seed = 7
np.random.seed(seed)

# Label encode Class
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# One Hot Encode
y_dummy = np_utils.to_categorical(encoded_Y)

# Deep Learnig Function
def deepml_model():
    # Model Creation
    deepml = Sequential()
    deepml.add(Dense(8, input_dim=60, activation='relu'))
    deepml.add(Dense(3, activation='softmax')) # Using softmax to ensure that the output values are in the range of 0 to 1 so that it can be used as predicted probabilities
    # Model Compilation
    #Here we are using the Adam gradient descent optimization with a logarithimic loss function called categorical_crossentropy
    deepml.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    return deepml

estimate = KerasClassifier(build_fn=deepml_model, epochs=20, batch_size=5, verbose=0)

k_fold = KFold(n_splits=10, shuffle=True, random_state=seed) #K-fold crossvalidation is used here with 10 splits

results = cross_val_score(estimate, X, y_dummy, cv=k_fold)
print("Model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
