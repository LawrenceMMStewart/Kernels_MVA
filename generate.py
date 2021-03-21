"""
Tuning parameters in the kernels
"""

import os
import pandas as pd
import pdb
import numpy as np
from models import * 
from kernels import * 
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ =="__main__":

    """
    For greg: to do debugging on KLR run something like this with whatever param you want

    python load.py --model KLR --K linear --regs 1e-3 --debug True
    """

    import argparse
    parser = argparse.ArgumentParser(description='Generate Prediction Files')
    parser.add_argument("--models",nargs='+',type=str,help="models for each dataset")
    parser.add_argument("--Ks", nargs='+',type = str,help ="kernel for each dataset" )
    parser.add_argument('--regs', nargs='+', help='reg value for each dataset', type = float)
    parser.add_argument("--params",nargs='+',type=float,help= "parameters e.g. l in gaussian kernel or degree of poly kernel")
    args = parser.parse_args()

    MODEL_MAP = {"KRR":KRR,"KLR":KLR,"KSVM":KSVM}
    KERNEL_MAP = {"linear":linear_kernel,
    "exp": lambda X,Y,p: exp_kernel(X,Y,l=p),
    "poly": lambda X,Y,p : polynomial_kernel(X,Y,deg=p)}

    preds=  []

    for i in range(3):

        model = MODEL_MAP[args.models[i]]
        kernel = KERNEL_MAP[args.Ks[i]]


        #load dataset into pandas
        Xtrain_path = os.path.join("data","Xtr%s_mat100.csv"%int(i))
        Ytrain_path = os.path.join("data","Ytr%s.csv"%int(i))
        dfXtrain = pd.read_csv(Xtrain_path,header=None,sep = " ")
        dfYtrain = pd.read_csv(Ytrain_path,header=0,sep = ",")

        Xtest_path = os.path.join("data","Xte%s_mat100.csv"%int(i))
        dfXtest = pd.read_csv(Xtest_path,header=None,sep = " ")


        # convert to numpy arrays
        Xtrain = dfXtrain.to_numpy()
        Ytrain = dfYtrain["Bound"].to_numpy()
        # convert Y to take values in {-1,1}
        Ytrain = (2*Ytrain - 1).reshape(-1,1)

        Xtest = dfXtest.to_numpy()


        #standardise data
        scaler = standardise(Xtrain)
        Xtrain_  = scaler.scale(Xtrain)
        Xtest_ = scaler.scale(Xtest)


        ker = lambda X,Y : kernel(X,Y,args.params[i])
        model_  = model(kernel = ker, reg=args.regs[i])

        model_.fit(Xtrain,Ytrain)
        Ypred = model_.predict(Xtest)
        Ypred[Ypred>=0] = 1.
        Ypred[Ypred<0] = 0.
        preds.append(Ypred)

    preds = np.vstack(preds).flatten().tolist()
    preds = [str(int(p)) for p in preds]
    ids = [str(i) for i in range(len(preds))]

    odf = pd.DataFrame({'Id':ids,'Bound':preds})
    with open('out.csv','w') as f:
        f.write(odf.to_csv(index=False))



