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



    # import argparse
    # parser = argparse.ArgumentParser(description='Generate Prediction Files')
    # parser.add_argument("--model",default="KRR",type=str,help="model for each dataset")
    # parser.add_argument('--reg', default=0.01, help='reg value for each dataset', type = float)
    # parser.add_argument("--K",default="spectrum",type = str,help = "kernel to use spectrum or mismatch")
    # parser.add_argument("--spec_size",default=6,type=int,help= "spectrum sizes")
    # args = parser.parse_args()

    MODEL_MAP = {"KRR":KRR,"KLR":KLR,"KSVM":KSVM}

    preds=  []
    model = MODEL_MAP["KLR"]
    KS = 8
    POS = all_seq(KS)
    print("generated all possible subsquences")


    d = generate_one_change(POS)
    print("generated all mismatch changes")


    for i in tqdm(range(3),desc="Dataset"):

        X,Y,Xtest = load_std(id=i)


        
        X_ = mismatch_embed_data(X,KS,POS,d)
        Xtest_ = mismatch_embed_data(Xtest,KS,POS,d)

        model_  = model(kernel = spec_kernel, reg=0.06)

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



