# Author: Ying Xu & Emma Westin

import random
import numpy as np
np.random.seed(3)
random.seed(1)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# adding argument for epoch (default value is 10), optimizer (default Adadelta) 
parser=argparse.ArgumentParser()
parser.add_argument('--e', nargs='?', const=10, type=int, default=10, help="epoch value")
parser.add_argument('--o', type=str, default='Adadelta', help="optimizer")
args=parser.parse_args()

dna_dict = {
        'A': [1, 0, 0, 0],
        'T': [0, 0, 0, 1],
        'G': [0, 0, 1, 0],
        'C': [0, 1, 0, 0],
        'D': [1, 0, 1, 1],
        'N': [1, 1, 1, 1],
        'S': [0, 1, 1, 0],
        'R': [1, 0, 1, 0]
    }

label_dict = {
        'IE': 0,
        'EI': 1,
        'N': 2
    }

# function to convert dna sequences to binary matrix
def seq2bin(dna_seq):
    dna_seq = dna_seq.strip()
    bin_mat = np.zeros((60,4))
    for i in range(len(dna_seq)):
        bin_mat[i, ] = dna_dict[dna_seq[i]]
    return bin_mat

# function to convert dna label to integers
def label2int(y_label):
    y_int = []
    for i in range(len(y_label)):
        y_int.append(label_dict[y_label[i]])
    return y_int

# function to build 4d array as model input
def reshape(x_train):
    num_train = len(x_train)
    x_train_mat = np.zeros((num_train, 60, 4, 1))
    for i in range(num_train):
        seq = x_train[i]
        x_train_mat[i,:,:,0] = seq2bin(seq)
    return x_train_mat

# build training and testing set
# 90% (2870 sequences) for training, 10% (320 sequences) for testing
df = pd.read_csv('splice.txt', sep=",", header=None)
df.columns = ['label', 'ID', 'sequence']
labels = df.as_matrix(columns=df.columns[:1])
sequences = df.as_matrix(columns=df.columns[2:3])
dfsize=df.shape[0]
indices = random.sample(range(dfsize), dfsize)
train_indices = indices[:2870]
test_indices = indices[2870:]
x_train = []
y_train = []
for i in train_indices:
    x_train.append(sequences[i,0])
    y_train.append(labels[i,0])
x_test = []
y_test = []
for i in test_indices:
    x_test.append(sequences[i, 0])
    y_test.append(labels[i, 0])

# define hyperparameters
batch_size = 1
num_classes = 3
epochs = args.e
rows, cols = 60, 4

# prepare training data input
x_train = reshape(x_train)
x_test = reshape(x_test)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = label2int(y_train)
y_test = label2int(y_test)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# model architecture
model = Sequential()
model.add(Conv2D(1, kernel_size=(3, 4),
                 activation='relu',
                 input_shape=(60,4,1)))
model.add(Flatten())
model.add(Dense(58, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.25)) # applying one dropout layer before output layer
model.add(Dense(num_classes, activation='softmax'))

# adam optimizer
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)

# RMSprop optimizer
rmsprop = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)

# SGD optimizer
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Adadelta optimizer
Adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

# compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=args.o,
              metrics=['accuracy'])

# train and validate model
model_history= model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Confusion Matrix for CNN prediction
pred_classes= model.predict_classes(x_test)
y_argmax=y_test.argmax(1)
print("Confusion matrix:\n{0}".format(metrics.confusion_matrix(y_argmax, pred_classes)))
# Precision and accuracy:
print("Classification report:\n{0}".format(metrics.classification_report(y_argmax, pred_classes)))
print("Classification accuracy: {0}".format(metrics.accuracy_score(y_argmax, pred_classes)))

# Plot for CNN accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('CNN Model of Splice Site Sequences Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

# Plot for CNN loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('CNN Model of Splice Site Sequences Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
