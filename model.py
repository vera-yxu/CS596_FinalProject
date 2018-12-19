from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
import numpy as np

# Pandas display in console
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

with open('splice.data', 'r') as infile:
    data = infile.readlines()
    data = [i.replace('\n', ' ').replace(' ', '').split(',') for i in data]
    df = pd.DataFrame(data)
    # Drop name column
    df = df.drop(columns=[1])

    enc_dict = {
        'A': np.array([1, 0, 0, 0]),
        'T': np.array([0, 0, 0, 1]),
        'G': np.array([0, 0, 1, 0]),
        'C': np.array([0, 1, 0, 0]),
        'D': np.array([1, 0, 1, 1]),
        'N': np.array([1, 1, 1, 1]),
        'S': np.array([0, 1, 1, 0]),
        'R': np.array([1, 0, 1, 0])
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
    print(df)
    df2 = pd.DataFrame(df[2].values.tolist())
    # Base pairs split into separate columns
    print(df2)

    # df.to_csv('data.csv')
    # df4 = pd.DataFrame(df[0])
    # df3 = df4.merge(df2)