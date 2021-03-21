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
from spectrum import *
from load import *

if __name__ =="__main__":



    import argparse
    parser = argparse.ArgumentParser(description='Generate Prediction Files')
    parser.add_argument("--models",nargs='+',type=str,help="models for each dataset")
    parser.add_argument('--regs', nargs='+', help='reg value for each dataset', type = float)
    parser.add_argument("--spec_sizes",nargs='+',type=int,help= "spectrum sizes")
    args = parser.parse_args()

    MODEL_MAP = {"KRR":KRR,"KLR":KLR,"KSVM":KSVM}

    preds=  []

    for i in tqdm(range(3),desc="Dataset"):

        X,Y,Xtest = load_std(id=i)
        model = MODEL_MAP[args.models[i]]
        KS = args.spec_sizes[i]
        POS = all_seq(KS)
        NO_S = len(POS)
        
        X_ = spec_embed_data(X,KS,NO_S,POS)
        Xtest_ = spec_embed_data(Xtest,KS,NO_S,POS)

        model_  = model(kernel = spec_kernel, reg=args.regs[i])

        model_.fit(X_,Y)
        Ypred = model_.predict(Xtest_)
        Ypred[Ypred>=0] = 1.
        Ypred[Ypred<0] = 0.
        preds.append(Ypred)

    preds = np.vstack(preds).flatten().tolist()
    preds = [str(int(p)) for p in preds]
    ids = [str(i) for i in range(len(preds))]

    odf = pd.DataFrame({'Id':ids,'Bound':preds})
    with open('out.csv','w') as f:
        f.write(odf.to_csv(index=False))



