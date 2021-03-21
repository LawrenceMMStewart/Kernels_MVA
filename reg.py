"""
Tuning regularization
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


if __name__ =="__main__":

    """
    For greg: to do debugging on KLR run something like this with whatever param you want

    python load.py --model KLR --K linear --regs 1e-3 --debug True
    """

    import argparse
    parser = argparse.ArgumentParser(description='Perform K fold cross validation for a model \
        on varied regularization parameters')
    parser.add_argument("--model",default = "KRR",type=str,help="model to train [KRR,KLR,KSVM]")
    parser.add_argument("--K", default = "linear",type = str,help ="kernel [linear,exp,poly]" )
    parser.add_argument('--regs', nargs='+', help='list of regularizations to access', type = float,
        required=True)
    parser.add_argument("--scale",type=float,default=1.0,help= "paramter l in gaussian kernel")
    parser.add_argument("--degree",type = int,default = 2,help = "parameter for degree on polynomial kernel")
    parser.add_argument("--data",type=int,default=0,help= "dataset to load")
    args = parser.parse_args()

    MODEL_MAP = {"KRR":KRR,"KLR":KLR,"KSVM":KSVM}
    KERNEL_MAP = {"linear":linear_kernel,
    "exp": lambda X,Y : exp_kernel(X,Y,args.scale),
    "poly": lambda X,Y : polynomial_kernel(X,Y,deg=args.degree)}

    model = MODEL_MAP[args.model]
    kernel = KERNEL_MAP[args.K]

    X_,Y,_ = load_100(id=args.data)


    reg_accs =  []
    for i in tqdm(range(len(args.regs)),desc= "Regularization"):
        reg = args.regs[i]
        model_  = model(kernel = kernel,reg=reg)
        #k fold cross validation to calculate train and test acc
        ktacc,keacc = KFoldXVAL(X_,Y,model_,k=5)
        reg_accs.append((ktacc,keacc))

    for i in range(len(args.regs)):
        print(f"{args.model} with {args.K} with reg = {args.regs[i]} --> train_acc = {reg_accs[i][0]}, eval_acc = {reg_accs[i][1]}")


    if len(args.regs)>1:
        plt.style.use('ggplot')
        plt.grid('on')
        max_id = np.argmax([r[1] for r in reg_accs])
        plt.semilogx(args.regs,[r[0] for r in reg_accs],label = "Train")
        plt.semilogx(args.regs,[r[1] for r in reg_accs],label = "Validation")
        plt.semilogx(args.regs[max_id],[r[1] for r in reg_accs][max_id],label = "Best",marker='D',color='g')
        plt.legend()
        plt.xlabel(r"$\gamma$")
        plt.ylabel("acc")
        plt.show()




