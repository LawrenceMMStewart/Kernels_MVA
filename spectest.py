"""
spectrum kernel
"""

import os
import pandas as pd
import pdb
import numpy as np
from models import * 
from kernels import * 
import matplotlib.pyplot as plt
from tqdm import tqdm
from load import *
from spectrum import *


if __name__ =="__main__":



    import argparse
    parser = argparse.ArgumentParser(description='Perform K fold cross validation for a model \
        on varied regularization parameters')
    parser.add_argument("--model",default = "KRR",type=str,help="model to train [KRR,KLR,KSVM]")
    parser.add_argument('--reg', default = 0.01, help='reg value', type = float)
    parser.add_argument("--spec_sizes",nargs='+',type=int,help= "sizes of the spectrum to evaluate",
        required=True)
    parser.add_argument("--K",default="spectrum",type = str,help = "kernel to use spectrum or mismatch")
    parser.add_argument("--data",type=str,default='0',help= "which dataset to evaluate performance on [0,1,2]")
    parser.add_argument("--save",default=None,type=str,help="if not none save plot with name ")
    args = parser.parse_args()

    MODEL_MAP = {"KRR":KRR,"KLR":KLR,"KSVM":KSVM}



    model = MODEL_MAP[args.model]
    kernel = spec_kernel
    
    X,Y,_ = load_std(id=args.data)


    k_accs = [] 
    for i in tqdm(range(len(args.spec_sizes)),desc= f"{args.K}"):
        #global variable of all possible sequences
        KS = args.spec_sizes[i]
        POS = all_seq(KS)
        NO_S = len(POS)
        
        if args.K =="spectrum":
            X_ = spec_embed_data(X,KS,POS)
        elif args.K =="mismatch":
            d = generate_one_change(POS)
            X_ = mismatch_embed_data(X,KS,POS,d)

        model_  = model(kernel = kernel, reg=args.reg)
        #k fold cross validation to calculate train and test acc
        ktacc,keacc = KFoldXVAL(X_,Y,model_,k=5)
        k_accs.append((ktacc,keacc))

        tqdm.write(f"Spectrum {KS} results in dimensionality = {X_.shape[1]}, with acc = {keacc}")



    if len(args.spec_sizes)>1:
        plt.style.use('ggplot')
        plt.grid('on')
        max_id= np.argmax([r[1] for r in k_accs])
       
            
        plt.plot([int(i) for i in args.spec_sizes],[r[0] for r in k_accs],label = "Train",marker= '.')
        plt.plot([int(i) for i in args.spec_sizes],[r[1] for r in k_accs],label = "Validation",marker= '.')
        plt.plot(args.spec_sizes[max_id],[r[1] for r in k_accs][max_id],label = "Best",marker='D',color='g')
        
     

        #---------
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("acc")
        if args.save is not None:
            plt.savefig("saved_plots/"+args.save)
        plt.show()




