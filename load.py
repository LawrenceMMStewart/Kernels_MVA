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


def calculate_acc(y,ypred):
    """
    calculate the accuracy
    """

    #threshold if a regression model
    ypred[ypred>=0]=1
    ypred[ypred<0] = -1

    return np.mean(ypred==y)


def KFoldXVAL(X,Y,model,k=5):
    """
    Performs K-fold cross validation on dataset

    Args:
    X : n x d data
    Y  : n x 1 labels
    kernel : function calculating the kernel matrix from X
    model : class with functions fit and predict to evaluate

    Returns:
    Acc : float [0,1]
    """
    assert len(X)==len(Y)
    n = len(X)
    assert k<n

    N =  n // k #size of datasets

    accs = []
    for i in tqdm(range(k),desc="Fold:"):
        
        eval_ids = np.arange(i*N,(i+1)*N)
        train_ids  = np.array( [i for i in range(n) if i not in eval_ids]  )

        Xtrain = X[train_ids]
        Xeval =  X[eval_ids]

        Ytrain = Y[train_ids]
        Yeval = Y[eval_ids]

        model.fit(Xtrain,Ytrain)
        predictions = model.predict(Xeval)
        fold_acc = calculate_acc(Yeval,predictions)
        accs.append(fold_acc)
        # print(f"Fold {i} obtained acc of {fold_acc}")

    return np.mean(accs)





if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Perform K fold cross validation for a model \
        on varied regularization parameters')
    parser.add_argument("--model",default = "KRR",type=str,help="model to train [KRR,KLR,KSVM]")
    parser.add_argument("--K", default = "linear",type = str,help ="kernel [linear,]" )
    parser.add_argument('--regs', nargs='+', help='list of regularizations to access', type = float,
        required=True)
    args = parser.parse_args()

    MODEL_MAP = {"KRR":KRR,"KLR":KLR,"KSVM":KSVM}
    KERNEL_MAP = {"linear":linear_kernel}

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


    # model =  KLR(kernel = linear_kernel,reg=9*1e-3)
    # model.fit(X_,Y,conv_plot = True)


    # # model  = KRR(kernel = linear_kernel,reg=0.01)
    # regs = [0.1**i for i in range(7)]


    reg_accs =  []
    for reg in args.regs:
        model_  = model(kernel = kernel,reg=reg)
        kacc = KFoldXVAL(X_,Y,model_,k=5)
        reg_accs.append(kacc)

    plt.style.use('ggplot')
    plt.grid('on')
    plt.semilogx(args.regs,reg_accs)
    plt.xlabel(r"$\gamma$")
    plt.ylabel("acc")
    plt.show()



