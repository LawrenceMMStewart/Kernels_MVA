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
    parser.add_argument("--K", default = "linear",type = str,help ="kernel [linear,exp,poly]" )
    parser.add_argument('--regs', nargs='+', help='list of regularizations to access', type = float,
        required=True)
    parser.add_argument("--scale",type=float,default=1.0,help= "paramter l in gaussian kernel")
    parser.add_argument("--degree",type = int,default = 2,help = "parameter for degree on polynomial kernel")
    args = parser.parse_args()

    MODEL_MAP = {"KRR":KRR,"KLR":KLR,"KSVM":KSVM}
    KERNEL_MAP = {"linear":linear_kernel,
    "exp": lambda X,Y : exp_kernel(X,Y,args.scale),
    "poly": lambda X,Y : polynomial_kernel(X,Y,deg=args.degree)}

    model = MODEL_MAP[args.model]
    kernel = KERNEL_MAP[args.K]


    #load dataset into pandas
    Xpath = os.path.join("data","Xtr0_mat100.csv")
    Ypath = os.path.join("data","Ytr0.csv")
    dfX = pd.read_csv(Xpath,header=None,sep = " ")
    dfY = pd.read_csv(Ypath,header=0,sep = ",")

    # convert to numpy arrays
    X = dfX.to_numpy()
    Y = dfY["Bound"].to_numpy()
    # convert Y to take values in {-1,1}
    Y = (2*Y - 1).reshape(-1,1)

    #standardise data
    scaler = standardise(X)
    X_  = scaler.scale(X)

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




