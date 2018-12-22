import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Creating function to calculate accuracy from confusion matrix
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return (diagonal_sum / sum_of_all_elements)*100
# Data Pre-processing
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
np.random.seed(101)
dataset = dataset.sample(frac=1).reset_index(drop=True)
i = pd.DataFrame(dataset)
seq = i.values

# Making Training and Test datasets
X = seq[0:, 0:60].astype(float) # features
Y = seq[0:, 60] # labels

# Set Seed
np.random.seed(7)

# Label encode Class
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# One Hot Encode
y_dummy = np_utils.to_categorical(encoded_Y)
X_train,X_test, y_train,y_test = train_test_split(X,y_dummy,test_size=0.2,random_state=0)

# Model Creation
deepml = Sequential()
deepml.add(Dense(10, input_dim=60, activation='relu'))
deepml.add(Dense(8,activation='relu'))
deepml.add(Dense(5,activation='relu'))
deepml.add(Dense(3, activation='softmax'))
# Model Compilation
deepml.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
deepml.summary()

#fitting the model and predicting
history = deepml.fit(X_train, y_train, epochs=200 ,validation_data=(X_test, y_test))
y_pred = deepml.predict(X_test)

y_test_class = np.argmax(y_test,axis=1) # convert encoded labels into classes
y_pred_class = np.argmax(y_pred,axis=1) # convert predicted labels into classes
#Accuracy of the predicted values
print(classification_report(y_test_class, y_pred_class)) # Precision , Recall, F1-Score & Support
cm = confusion_matrix(y_test_class, y_pred_class)
print(cm)
print("Model Accuracy: {} %" .format(accuracy(cm)))

training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();


