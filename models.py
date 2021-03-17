import numpy as np
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
		assert reg>0 , "Enter a positive reg param"
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

	def predict(self,X2):
		assert self.alpha is not None , "Fit model before predictions"
		#distance between points
		K = self.kernel(X2,self.X)
		return K@self.alpha



def solve_WKRR(K,W,z,reg):
	"""
	solves weighted kernel ridge regression
	(used for KLR)

	slide 103 of cours
	"""
	#matrix sqrt of W
	WR = W**(0.5)
	n = K.shape[0]
	A = WR*K*WR + n*reg*np.eye(n)
	b = WR*z
	alpha = WR * np.linalg.solve(A,b)
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
	min_alpha 1/n sum log(1 + exp{ -y_i K @alpha _i   }) + reg/2 ||alpha||_K

	solve using WKRR until convergence (slide 114 of class)
	"""

	def __init__(self,kernel = None, reg = 0.1):
		self.kernel = kernel
		self.reg = reg
		assert reg>0 , "Enter a positive reg param"
		self.X = None
		self.alpha = None


	def fit(self,X,Y,tol = 1e-5):
		assert self.kernel is not None , "Please input a valid kernel"
		n = X.shape[0]
		self.alpha = np.zeros((n,1))
		self.prev_alpha = np.ones((n,1))*tol
		self.X = X
		K = self.kernel(X,X)
	
		while np.linalg.norm(self.alpha-self.prev_alpha)>tol:
			m = K@ self.alpha
			W = sig(m)*sig(-m)
			z = m + Y / sig(-Y*m)

			self.prev_alpha = self.alpha
			self.alpha = solve_WKRR(K,W,z,self.reg)



	def predict(self,X2):
		assert self.alpha is not None , "Fit model before predictions"
		#distance between points
		K = self.kernel(X2,self.X)
		return K@self.alpha

