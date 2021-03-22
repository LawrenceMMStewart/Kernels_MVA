from itertools import product
import pandas as pd
import os
from load import load_std
import numpy as np
from copy import copy

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

def generate_one_change(pos):
    """
    create a dictionary which given a substring
    returns the indexes of all substrings that 
    are identical except for one change
    """
    ops =['A','C','T','G'] 
    d = {}
    #initialise the dictionary
    for sub in pos:
        d[sub] = []
    #list positions of one-change substrings
    for sub in pos:

        s = list(sub)
        for i,schar in enumerate(s):
            for newchar in ops:
                if schar!= newchar:
                    new_sub  = copy(s)
                    new_sub[i] = newchar
                    new_sub = tuple(new_sub)
                    d[sub] = d[sub]+ [pos.index(new_sub)]

    return d 


def spec_embed(s,k,pos):
    """
    embed a string s using spectrum RKHS
    """
    no_s  = len(pos)
    subs = all_ngrams(s,k)
    vec = np.zeros((no_s))
    for sub in subs:
        ind = pos.index(sub)
        vec[ind]+=1
    # return vec.reshape(1,-1)
    return vec

def spec_embed_data(X,k,pos):
    """
    embed a data matrix of DNA strings
    """
    no_s = len(pos)
    n = X.shape[0]
    phi = np.zeros((n,no_s))
    for i in range(n):
        phi[i] = spec_embed(X[i],k,pos)
    return phi



def mismatch_embed(s,k,pos,change_dic):
    """
    embed a string using the mis match-kernel
    allowing 1 mismatch (m=1)
    """
    no_s = len(pos)
    subs = all_ngrams(s,k)
    vec = np.zeros((no_s))
    for sub in subs:
        ind = pos.index(sub)
        vec[ind]+=1
        #1 mismatch 
        for ind in change_dic[sub]:
            vec[ind]+=1
    return vec

def mismatch_embed_data(X,k,pos,change_dic):
    no_s = len(pos)
    n = X.shape[0]
    phi = np.zeros((n,no_s))
    for i in range(n):
        phi[i] = mismatch_embed(X[i],k,pos,change_dic)
    return phi





# def CreateHistogramMismatchSeq(Seq,AllCombinList,n):
#     '''
#     Create the embedding that allows to compute the mismatch kernel: histogram of all the subsequences of length n
#     in the sequence. This time allows one mismatch.
#     Param: @Seq: (str) DNA sequence containing only the letter A,C,G,T
#     @n: (int) length of the subsequences considered
#     @AllCombinList: (list) a list containing all the possible combination of length n that we can compute using the letters
#     A C G T
#     Return: value : np.array contains the representation of the sequence as an array
#     '''
#     letters = ['A','C','G','T']
#     decompose_seq= ngrams(Seq,n)
#     value = np.zeros([len(AllCombinList),])
#     for ngram in decompose_seq:
#         index_ngram = AllCombinList.index(ngram)
#         value[index_ngram] = value[index_ngram]+1
#         copy_ngram = list(ngram)
#         for ind,cur_letter in enumerate(copy_ngram):
#             for letter in letters:
#                 if letter!=cur_letter:
#                     new_ngram = list(copy_ngram)
#                     new_ngram[ind]= letter
#                     mismatch_ngram = tuple(new_ngram)
#                     index_ngram = AllCombinList.index(mismatch_ngram)
#                     value[index_ngram] = value[index_ngram]+0.1
#     return value


def spec_kernel(X1,X2):
    return X1 @ X2.T



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Perform K fold cross validation for a model \
        on varied regularization parameters')

    parser.add_argument("--k", default = 3,type = int,help =" size of substrings in spectrum" )
    parser.add_argument("--K",default="spectrum",type = str,help = "kernel to use spectrum or mismatch")
    args = parser.parse_args()

    #global variable of all possible sequences
    KS = args.k
    POS = all_seq(KS)
    # NO_S = len(POS)
    X,Y,Xtest = load_std(id=0)

    if args.K =="spectrum":
        eX = spec_embed_data(X,KS,POS)
        eXtest = spec_embed_data(Xtest,KS,POS)

    elif args.K == 'mismatch':
        d = generate_one_change(POS)
        eX = mismatch_embed_data(X,KS,POS,d)
        eXtest =mismatch_embed_data(Xtest,KS,POS,d)

    import pdb; pdb.set_trace()


