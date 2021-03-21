from itertools import product
import pandas as pd
import os
from load import load_std
import numpy as np

def all_seq(k):
    """
    Generate all possible sequences
    of size n from the DNA characters
    """
    ops =['A','C','T','G']
    return list(product(ops,repeat=k))


def all_ngrams(s,k):
    """
    return all ngrams found in a string
    of size n
    """
    return list(zip(*[s[i:] for i in range(k)]))

def spec_embed(s,k,no_s,pos):
    """
    embed a string s using spectrum RKHS
    """
    subs = all_ngrams(s,k)
    vec = np.zeros((no_s))
    for sub in subs:
        ind = pos.index(sub)
        vec[ind]+=1
    return vec.reshape(1,-1)

def spec_embed_data(X,k,no_s,pos):
    n = X.shape[0]
    phi = np.zeros((n,no_s))
    for i in range(n):
        phi[i] = spec_embed(X[i],k,no_s,pos)
    return phi


def spec_kernel(X1,X2):
    return X1 @ X2.T



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Perform K fold cross validation for a model \
        on varied regularization parameters')

    parser.add_argument("--k", default = 3,type = int,help =" size of substrings in spectrum" )
    args = parser.parse_args()

    #global variable of all possible sequences
    KS = args.k
    POS = all_seq(KS)
    NO_S = len(POS)

    X,Y,Xtest = load_std(id=0)

    eX = spec_embed_data(X,KS,NO_S,POS)
    eXtest = spec_embed_data(Xtest,KS,NO_S,POS)

