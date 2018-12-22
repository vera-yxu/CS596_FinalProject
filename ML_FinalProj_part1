
"""
Created on Tue Dec 18 09:51:36 2018

@author: Will
"""
#Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt

#import data
df = pd.read_csv("splice.data",
    names=["Class", "Instance", "Sequence"])

#df.head()#checking out data
#df.describe()#check description

Y = df['Class'] #checking dataframe "class"
Y.groupby(Y).count()  
le = LabelEncoder()
le.fit(Y)
# record the label encoder mapping
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
y = le.transform(Y)

#le_name_mapping #look at the maps
#y.shape #look at y shape
#df['Instance'][:20] #chekcing out the instances.

def find_letters(row):
    return set(row.strip())

set_series = df['Sequence'].apply(find_letters)

"""Exploring the dataset"""
#set.union(*set_series)#find unique, show unique
#set_series[set_series.str.contains('N', regex=False)]#looking at other letters
#df.loc[107].Sequence.strip()
#df['Instance_Prefix'] = df['Instance'].str.extract("(.*)-(\w*)-(\d*)")[0]
#df['Instance_Donor'] = df['Instance'].str.extract("(.*)-(\w*)-(\d*)")[1]
#df['Instance_Number'] = df['Instance'].str.extract("(.*)-(\w*)-(\d*)")[2]
#df['Instance_Donor'].unique()
#array(['DONOR', 'ACCEPTOR', 'NEG'], dtype=object)
#donor_one_hot_df = pd.get_dummies(df['Instance_Donor'])
#donor_one_hot_df.sample(10, random_state=0)


df['Sequence'].str.strip().map(len).unique() #checking The dataset claims that every row has 60 characters (30 before and 30 after the possible splice. Let’s check to make sure that’s true.

"""Modifiying the Sequence"""

letter_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'D': 4, 'N': 5, 'R': 6, 'S': 7}

#convert
def convert_base_to_num(bases):
    return [letter_mapping[letter] for letter in bases.strip()]

df['Sequence_list'] = df['Sequence'].apply(convert_base_to_num)
#df['Sequence_list'].head() #converted the letters to integers. 

X_sequence = np.array(df['Sequence_list'].values.tolist())#split those lists so that each individual integer is in its own column. To do that we’ll convert the pandas series into an ndarray. 

#add to list
encoder = OneHotEncoder(n_values=len(letter_mapping))
X_encoded = encoder.fit_transform(X_sequence)
X_one_hot = X_encoded.toarray()

"""Save"""
X = X_one_hot
np.save('dna_cleaner.npy', X)
np.save('dna_clean_labels', y)
