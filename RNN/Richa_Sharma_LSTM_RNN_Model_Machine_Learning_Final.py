# Name: Richa Sharma
# LSTM RNN Machine Learning Final Project


# importing numpy, random for shuffle and matplotlib.pyplot
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

NUM_EPOCHS=100

# Defining a function get lines which reads my data file and extracts lines from my file,
# to extract the DNA sequences and the labels.
# The file splice.data has 3 columns in each line extracted, which are separated by commas and random number spaces.
# This function extracts each row as a single string separated by commas, and it makes a list of all the rows in the
# the file.
# Also, the function shuffles all the rows of the file before giving the output, because of which we get randomized data set.

def get_lines(splice_data_file):
    my_spice_data_file = open(splice_data_file)
    lines_my_splice_data = my_spice_data_file.readlines()
    lines_my_splice_data_no_space = [line.replace(' ', '') for line in lines_my_splice_data]
    shuffle(lines_my_splice_data_no_space)
    my_spice_data_file.close()
    return lines_my_splice_data_no_space


#print(get_lines("splice.txt"))

# Defining a function extract_dna_string to get the DNA sequence out of our rows of data.
# we thus use the output of get_lines function above and extract just the DNA sequences out of it.
def extract_dna_string(lines_my_splice_data_no_space):
    dna_strings_list = []
    for i in range(0,len(lines_my_splice_data_no_space)):
        dna_line = lines_my_splice_data_no_space[i]
        words_in_line = dna_line.split(",")
        #print(words_in_line)
        dna_string = words_in_line[2]
        dna_string = dna_string.strip()
        dna_strings_list.append(dna_string)
    return dna_strings_list


lines_my_splice_data_no_space = get_lines("splice.txt")
dna_strings_list = extract_dna_string(lines_my_splice_data_no_space)
#print(dna_strings_list)

# Defining a function dna_to_vector which converts the string of DNA in to a series of numbers
# depending upon the DNA base character. The function outputs a converted string of numbers for each
# of our DNA sequences obtained in the previous function extract_DNA_string.
def dna_to_vector(dna_strings_list):
    dna_string_conversion = {'A':1, 'T':2, 'G':3, 'C':4, 'D':5, 'N':6, 'S':7, 'R':8}
    list_of_list_dna_vector = []
    for i in range(0, len(dna_strings_list)):
        dna_string = dna_strings_list[i]
        converted_dna = []
        for base in dna_string:
            converted_dna.append(dna_string_conversion.get(base, base))
        concatenated_string = ''
        for element in converted_dna:
            concatenated_string += str(element)
        list_of_list_dna_vector.append(concatenated_string)
    return list_of_list_dna_vector


list_of_list_dna_vector = (dna_to_vector(dna_strings_list))
#print(list_of_list_dna_vector)

# Defining a function get_input_array to shape our data in to an array for our LSTM RNN model.
# It has an input parameter k which defines the length of the kmer which is going to be the feature length
# for our input data. The other parameter takes the input of the list of DNA strings that are converted to numbers,
# which is the output of the previous function dna_to_vector.
# I used the k parameter because I wanted to test my LSTM model below for different values of Kmer length (3, 5, 6).
def get_input_array(list_of_list_dna_vector, k):
    ret_list = []
    for element in list_of_list_dna_vector:
        sample = np.array(list(element))
        sample_reshape = sample.reshape(-1, k)
        ret_list.append(sample_reshape)
    ret = np.array(ret_list)
    dna_input = ret.astype(int)
    return dna_input

# calling the get_input_array function to get our data in X_data variable.


x_data = (get_input_array(list_of_list_dna_vector, 5))

# Making training and testing data sets by dividing the data in to 80:20 ratio
# for training and testing data.
x_train = x_data[:2552]
x_test = x_data[2552:]
#print(x_train.shape)
#print(x_train[0])
#print(x_train[1])
#print(x_test.shape)

# Defining a function extract_y_labels to get the labels for our data.
# It takes the input of the output of the first function get lines to give the output as a vector
# which contains all the classification labels for our data.
# These use the input from firs function get lines, hence the labels still correspond to the randomised
# DNA sequences correctly.
def extract_y_labels(lines_my_splice_data_no_space):
    y_label_list = []
    for i in range(0,len(lines_my_splice_data_no_space)):
        dna_line = lines_my_splice_data_no_space[i]
        words_in_line = dna_line.split(",")
        y_label = words_in_line[0]
        y_label_list.append(y_label)
    y_label_conversion = {'EI': 0, 'IE': 1, 'N': 2}
    y_labels = [y_label_conversion[x] for x in y_label_list]
    y_label_array = np.array(y_labels)
    y_label_array_reshaped = y_label_array.reshape(3190,1)
    return y_label_array_reshaped

# calling the function extract_y_labels to get the y_label vector data


y_data = extract_y_labels(lines_my_splice_data_no_space)

# Dividing the y label data in to training and testing label data in ration 80:20 for training and testing.
y_train = y_data[:2552]
y_test = y_data[2552:]
#print(y_train)
#print(y_train.shape)


# Constructing the LSTM RNN model using the Keras API using TensorFlow in backend.
# Making a sequential model
# Adding two hidden LSTM layers and two dense layers
# Using the activation functions tanh, sigmoid and softmax activation functions for these
# layers respectively.
# Using Adam optimizer and defining the learning rate through lr
# compiling the model using model.compile.
# used the loss function as sparse_categorical cross entropy and
# setting the accuracy metrics to accuracy which chooses the correct accuracy function as
# given by the loss function.
# using model.fit to train and test our model and define the number of epochs to be used.


model = Sequential()
model.add(LSTM(16, input_shape=(x_train.shape[1:]), activation='tanh', return_sequences=True))

model.add(LSTM(32, activation='sigmoid'))

model.add(Dense(8,activation='sigmoid'))

model.add(Dense(3, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=3e-3)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, validation_data=(x_test, y_test))

# Plotting the loss vs epoch plot using matplotlib.pyplot

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
print("Close the plot to continue...")
plt.show()

# Plotting the accuracy vs epoch plot using matplotlib.pyplot

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
print("Close the plot to continue...")
plt.show()
print("Finished")