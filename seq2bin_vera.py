
# Convert DNA sequence to binary matrix

import numpy as np
import pandas as pd

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

def seq2bin(dna_seq):
    dna_seq = dna_seq.strip()
    bin_mat = np.zeros((60,4))
    for i in range(len(dna_seq)):
        bin_mat[i, ] = dna_dict[dna_seq[i]]
    return bin_mat

'''
# test function
dna_seq='CCAGCTGCATCACAGGAGGCCAGCGAGCAGGTCTGTTCCAAGGGCCTTCGAGCCAGTCTG'
print(seq2bin(dna_seq))
'''

df = pd.read_csv('splice.txt', sep=", ", header=None)
df.columns = ['label', 'ID', 'sequence']
labels = df.as_matrix(columns=df.columns[:1])
sequences = df.as_matrix(columns=df.columns[2:3])
# print(sequences) all sequences
# print(sequences[0,0]) first sequence
# print(sequences[1,0]) second sequence

'''
# convert all sequences into binary 
for i in range(len(sequences)):
    sequences[i,0]=seq2bin(sequences[i,0])
'''


