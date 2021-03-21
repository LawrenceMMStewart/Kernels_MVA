import pandas as pd 
import os

class standardise():

    def __init__(self,X,method='max_min'):
        """
        fit standardiser to data
        """
        if method == "max_min":
            self.Max = X.max(axis=0)
            self.Min = X.min(axis=0)
        else:
            raise ValueError("Please enter a valid method")

    def scale(self,X):
        """
        scale a data matrix X
        """

        Xhat = (X - self.Min) / (self.Max - self.Min)
        return Xhat



def load_100(id=0):
    """
    load data file (mat)
    and standardise
    """
    #load dataset into pandas
    Xpath = os.path.join("data","Xtr%s_mat100.csv"%id)
    Ypath = os.path.join("data","Ytr%s.csv"%id)
    Xtpath = os.path.join("data","Xte%s_mat100.csv"%id)
    dfX = pd.read_csv(Xpath,header=None,sep = " ")
    dfY = pd.read_csv(Ypath,header=0,sep = ",")
    dfXt =  pd.read_csv(Xtpath,header=None,sep = " ")

    # convert to numpy arrays
    X = dfX.to_numpy()
    Y = dfY["Bound"].to_numpy()
    # convert Y to take values in {-1,1}
    Y = (2*Y - 1).reshape(-1,1)

    Xt = dfXt.to_numpy()

    #standardise data
    scaler = standardise(X)
    X_  = scaler.scale(X)
    Xt_ = scaler.scale(Xt)

    return X_,Y,Xt


def load_std(id=0):
    #load dataset into pandas
    Xpath = os.path.join("data",f"Xtr{id}.csv")
    Xtpath = os.path.join("data",f"Xte{id}.csv")
    Ypath = os.path.join("data",f"Ytr{id}.csv")
    dfX = pd.read_csv(Xpath,header=0,sep = ",")
    dfY = pd.read_csv(Ypath,header=0,sep = ",")
    dfXt = pd.read_csv(Xtpath,header=0,sep = ",")

    # convert to numpy arrays
    X = dfX['seq'].to_numpy()
    Y = dfY["Bound"].to_numpy()
    Y = (2*Y - 1).reshape(-1,1)
    Xt = dfXt['seq'].to_numpy()

    return X,Y,Xt