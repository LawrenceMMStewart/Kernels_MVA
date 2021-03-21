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
from load import *
"""

- A suggested plan is

   a) implement a linear classifier to use the _mat100.csv  files (e.g., logistic regression). You can also start with a ridge regression estimator 
   used as a classifier (regression on labels -1, or +1), since this is very easy to implement.

   b) move to a nonlinear classifier by using a Gaussian kernel (still using the *_mat100.csv files).  
   Kernel ridge regression is a good candidate, but then you may move to a support vector machine (use a QP solver). 
   Wny starting with kernel ridge regression => because it can be implemented in a few lines of code.


   c) then start working on the raw sequences. Time to design a good kernel for this data!


- You can find examples of practical sessions on kernels here, which were designed by Romain Menegaux,
 a student of JP : http://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/course/2019ammi/index.html
- read the rules carefully. In particular, the DIY rule.  You can also form teams of 3 students. 
- the goal is not necessarily to introduce some competition among students. If you implement the right baselines, you are going to get a good grade!  
- lectures on kernels for biological sequences are available on the course website (see Bonus at the bottom).
- Good luck!
"""

if __name__ =="__main__":

    """
    For greg: to do debugging on KLR run something like this with whatever param you want

    python load.py --model KLR --K linear --regs 1e-3 --debug True
    """

    import argparse
    parser = argparse.ArgumentParser(description='Perform K fold cross validation for a model \
        on varied regularization parameters')
    parser.add_argument("--model",default = "KRR",type=str,help="model to train [KRR,KLR,KSVM]")
    parser.add_argument("--K", default = "poly",type = str,help ="kernel [exp,poly]" )
    parser.add_argument('--reg', default = 0.01, help='reg value', type = float)
    parser.add_argument("--params",nargs='+',type=float,help= "parameters e.g. l in gaussian kernel or degree of poly kernel",
        required=True)
    parser.add_argument("--data",type=str,default='0',help= "which dataset to evaluate performance on [0,1,2,all]")
    parser.add_argument("--save",default=None,type=str,help="if not none save plot with name ")
    args = parser.parse_args()

    MODEL_MAP = {"KRR":KRR,"KLR":KLR,"KSVM":KSVM}
    KERNEL_MAP = {"linear":linear_kernel,
    "exp": lambda X,Y,p: exp_kernel(X,Y,l=p),
    "poly": lambda X,Y,p : polynomial_kernel(X,Y,deg=p)}

    names = {'exp':'scale','poly':'degree'}

    model = MODEL_MAP[args.model]
    kernel = KERNEL_MAP[args.K]
    
    X_,Y,_ = load_100(id=args.data)

    p_accs =  []
    for i in tqdm(range(len(args.params)),desc= "params"):
        ker = lambda X,Y : kernel(X,Y,args.params[i])
        model_  = model(kernel = ker, reg=args.reg)
        #k fold cross validation to calculate train and test acc
        ktacc,keacc = KFoldXVAL(X_,Y,model_,k=5)
        p_accs.append((ktacc,keacc))



    for i in range(len(args.params)):
        print(f"{args.model} with {args.K} with {names[args.K]} = {args.params[i]} --> train_acc = {p_accs[i][0]}, eval_acc = {p_accs[i][1]}")


    if len(args.params)>1:
        plt.style.use('ggplot')
        plt.grid('on')
        max_id= np.argmax([r[1] for r in p_accs])
        if args.K == 'poly':
            
            plt.plot([int(i) for i in args.params],[r[0] for r in p_accs],label = "Train",marker= '.')
            plt.plot([int(i) for i in args.params],[r[1] for r in p_accs],label = "Validation",marker= '.')
            plt.plot(args.params[max_id],[r[1] for r in p_accs][max_id],label = "Best",marker='D',color='g')
            
     
        elif args.K == "exp":
            plt.semilogx(args.params,[r[0] for r in p_accs],label = "Train",marker= '.')
            plt.semilogx(args.params,[r[1] for r in p_accs],label = "Validation",marker= '.')
            plt.semilogx(args.params[max_id],[r[1] for r in p_accs][max_id],label = "Best",marker='x',color='g')

        #---------
        plt.legend()
        plt.xlabel(f"{names[args.K]}")
        plt.ylabel("acc")
        if args.save is not None:
            plt.savefig("saved_plots/"+args.save)
        plt.show()




