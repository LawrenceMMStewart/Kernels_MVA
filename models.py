import numpy as np
from scipy.linalg import block_diag
# import cvxpy as cp
# from cvxopt import matrix
# from cvxopt import solvers
import cvxpy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
"""
Kernel Ridge Regression
"""
class KRR():
	"""
	Class for Kernel ridge regression
	predicting real labels

	alpha = (K + n * lambda * I)^-1 y
	"""

	def __init__(self,kernel = None,reg = 0.1):
		"""
		initialise kernel ridge regression
		"""
		self.reg = reg 
		assert reg>=0 , "Enter a positive reg param"
		self.alpha = None 
		self.kernel = kernel
		self.X = None


	def fit(self,X,Y):
		#reset weights
		assert self.kernel is not None , "Please input a valid kernel"
		self.alpha = None
		self.X = X
		K  = self.kernel(X,X)
		n = K.shape[0]
		self.alpha = np.linalg.solve(K +self.reg*n*np.eye(n),Y)
		#save kernel matrix for train data
		self.Kx = K

	def predict(self,X2=None):
		assert self.alpha is not None , "Fit model before predictions"

		if X2 is not None:
			#distance between points
			K = self.kernel(X2,self.X)
			return K@self.alpha
		else:
			return self.Kx @ self.alpha



#def solve_WKRR(K,W,z,reg):
#	"""
#	solves weighted kernel ridge regression
#	(used for KLR)
#
#	slide 153 of cours
#	"""
#	#matrix sqrt of W
#	WR = W**(0.5)
#	n = K.shape[0]
#	A = K*W + n*reg*np.eye(n)
#	b = WR*z
#	alpha = WR * np.linalg.solve(A,b)
#	return alpha

def solve_WKRR(K,W,z,reg):
    """
    solves weighted kernel ridge regression
    (used for KLR)
    slide 153 of cours
    """
    n = K.shape[0]
    WR = np.diag(np.sqrt(W.reshape(-1)))
    WR_inv = np.diag(1 / np.sqrt(W.reshape(-1)))
    alpha = np.linalg.solve((WR @ K @ WR + reg * n * np.eye(n)) @ WR_inv, WR @ z)
    return alpha

def sig(x):
    "Numerically stable sigmoid function."
    l0  = np.where(x<0)[0]
    geq0 = np.where(x>=0)[0]
    s = x.copy()

    z = np.exp(x[l0])
    z_ = np.exp(-x[geq0])
    if len(l0)>0:
    	s[l0] = z/(1+z)
    if len(geq0)>0:
    	s[geq0] = 1 / (1+z_)

    return s 

 


class KLR():
	"""
	Class for kernel logistic regression
	solving
	min_alpha 1/n sum log(1 + exp{ -y_i (K @alpha)_i   }) + reg/2 ||alpha||_K

	solve using WKRR until convergence (slide 114 of class)
	"""
	def __init__(self,kernel = None, reg = 0.1):
		self.kernel = kernel
		self.reg = reg
		assert reg>=0 , "Enter a positive reg param"
		self.X = None
		self.alpha = None
	
	def fit(self,X,Y,tol = 1e-4, max_iters = 50, conv_plot = False):
		assert self.kernel is not None , "Please input a valid kernel"
		n = X.shape[0]
		self.alpha = np.zeros((n,1))
		self.prev_alpha = np.ones((n,1))*2*tol
		self.X = X
		K = self.kernel(X,X)
		#save train set kernel
		self.Kx = K

		#difference in norm of alpha values
		gaps = []
		i = 0
		gap  = 2*tol
		while gap>tol:
			gap  = np.abs(self.alpha - self.prev_alpha).max()
			gaps.append(gap)
			i += 1
			if i > max_iters:
				break

			##update weights 
			m = K@ self.alpha
			W = sig(Y*m)*sig(-Y*m)
			P = -sig(-Y*m)
			z = m - P*Y/W

			self.prev_alpha = self.alpha
			self.alpha = solve_WKRR(K,W,z,self.reg)

		if conv_plot:
			xs = [i+1 for i in range(len(gaps))]
			plt.plot(xs,gaps)
			plt.title(f"max_iters = {max_iters}")
			plt.xlabel("Iteration")
			plt.ylabel(r"$|| \alpha_n - \alpha_{n-1}||_\infty$")
			plt.show()
			


	def predict(self,X2=None):
		assert self.alpha is not None , "Fit model before predictions"

		if X2 is not None:
			#distance between points
			K = self.kernel(X2,self.X)
			return K@self.alpha
		else:
			return self.Kx @ self.alpha



class KSVM():
	"""
	Solve Kernel Support Vector Machine QP (C-form)
	see slide 165 of class
	"""

	def __init__(self,kernel = None,reg = 0.1):
		"""
		initialise kernel ridge regression
		"""
		self.reg = reg 
		assert reg>0 , "Enter a positive reg param"
		self.alpha = None 
		self.kernel = kernel
		self.X = None


	def fit(self,X,Y):
		#reset weights
		assert self.kernel is not None , "Please input a valid kernel"
		n = X.shape[0]
		self.X = X 
		K = self.kernel(X,X)
		#save train set kernel
		self.Kx = K

		alpha = cp.Variable((n,1))
		problem  = cp.Problem(cp.Minimize(-2*alpha.T@Y + cp.quad_form(alpha,K)),
			[
			2*n*self.reg*cp.multiply(Y,alpha)<=1,
			cp.multiply(Y,alpha)>=0
			])
		problem.solve()

		self.alpha= alpha.value



	def predict(self,X2=None):
		assert self.alpha is not None , "Fit model before predictions"

		if X2 is not None:
			#distance between points
			K = self.kernel(X2,self.X)
			return K@self.alpha
		else:
			return self.Kx @ self.alpha



# class standardise():

#     def __init__(self,X,method='max_min'):
#         """
#         fit standardiser to data
#         """
#         if method == "max_min":
#             self.Max = X.max(axis=0)
#             self.Min = X.min(axis=0)
#         else:
#             raise ValueError("Please enter a valid method")

#     def scale(self,X):
#         """
#         scale a data matrix X
#         """

#         Xhat = (X - self.Min) / (self.Max - self.Min)
#         return Xhat


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

    eval_accs = []
    train_accs = []
    for i in tqdm(range(k),desc="Fold:"):
        
        eval_ids = np.arange(i*N,(i+1)*N)
        train_ids  = np.array( [i for i in range(n) if i not in eval_ids]  )

        Xtrain = X[train_ids]
        Xeval =  X[eval_ids]

        Ytrain = Y[train_ids]
        Yeval = Y[eval_ids]

        model.fit(Xtrain,Ytrain)


        eval_predictions = model.predict(Xeval)
        eval_fold_acc = calculate_acc(Yeval,eval_predictions)
        eval_accs.append(eval_fold_acc)

        train_predictions = model.predict()
        train_fold_acc = calculate_acc(Ytrain,train_predictions)
        train_accs.append(train_fold_acc)
        # print(f"Fold {i} obtained acc of {fold_acc}")

    tacc = np.mean(train_accs)
    eacc = np.mean(eval_accs)


    return tacc,eacc

