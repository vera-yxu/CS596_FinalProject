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
        'A': np.array([1,0,0,0]),
        'T': np.array([0,0,0,1]),
        'G': np.array([0,0,1,0]),
        'C': np.array([0,1,0,0]),
        'D': np.array([1,0,1,1]),
        'N': np.array([1,1,1,1]),
        'S': np.array([0,1,1,0]),
        'R': np.array([1,0,1,0])
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

def forest_predictor(file_path, classcolumn, **kwargs):
    with open(file_path, 'r') as infile:
        # data = [line.replace('\n', '').split(',') for line in infile]
        # header_exists = kwargs.get('header', False)
        # if header_exists:
        #     x = data[1:]
        #     y = data[1:]
        data = infile.readlines()
        data = [i.replace('\n', ' ').replace(' ', '').split(',') for i in data]
        x = [line[:-1] for line in data]
        y = [line[classcolumn] for line in data]

        save_exists = kwargs.get('save', False)
        if save_exists in file_path:
            with open(save_exists, 'rb') as infile:
                x1 = pickle.load(infile)
                print(x1)
        else:
            clf = RandomForestClassifier(n_estimators=500)
            clf = clf.fit(x, y)
            with open(save_exists, 'wb') as outfile:
                pickle.dump(x, outfile)
        test_exists = kwargs.get('test', False)
        if test_exists:
            print(clf.predict(test_exists))
    return clf


# if __name__ == '__main__':
    # clf = forest_predictor('splice.data', 0, header=True, save='random_forest.p',
    #                        test=[[15, 0], [18, 60000], [80, 30000]])
    # # Should print ['0', '1', '1'] and return the classifier so that feature_importances_ can be printed
    # print(clf)