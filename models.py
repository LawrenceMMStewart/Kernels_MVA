import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
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

	slide 153 of cours
	"""
	#matrix sqrt of W
	WR = W**(0.5)
	n = K.shape[0]
	A = K*W + n*reg*np.eye(n)
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

    # if x >= 0:
    #     z = exp(-x)
    #     return 1 / (1 + z)
    # else:
    #     # if x is less than zero then z will be small, denom can't be
    #     # zero because it's 1+z.
    #     z = exp(x)
    #     return z / (1 + z)


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
		assert reg>0 , "Enter a positive reg param"
		self.X = None
		self.alpha = None
	
	def fit(self,X,Y,tol = 1e-2, max_iters = 50, conv_plot = False):
		assert self.kernel is not None , "Please input a valid kernel"
		n = X.shape[0]
		self.alpha = np.zeros((n,1))
		self.prev_alpha = np.ones((n,1))
		self.X = X
		K = self.kernel(X,X)

		#difference in norm of alpha values
		gaps = []
		i = 0
		gap  = 2*tol
		while gap>tol:
			gap  = np.linalg.norm(self.alpha-self.prev_alpha)
			gaps.append(gap)
			i += 1
			if i > max_iters:
				break

			#update weights
			m = K@ self.alpha
			W = sig(Y*m)*sig(-Y*m)
			P = -sig(-Y*m)
			z = m - P*Y/W

			# #delete this debugging line
			# if conv_plot:
			# 	z2 = m + Y / sig(Y*m)

			self.prev_alpha = self.alpha
			self.alpha = solve_WKRR(K,W,z,self.reg)

		if conv_plot:
			xs = [i+1 for i in range(len(gaps))]
			plt.plot(xs,gaps)
			plt.xlabel("Iteration")
			plt.ylabel(r"$|| \alpha_n - \alpha_{n-1}||_2$")
			plt.show()
			


	def predict(self,X2):
		assert self.alpha is not None , "Fit model before predictions"
		#distance between points
		K = self.kernel(X2,self.X)
		return K@self.alpha



class KSVM():
	"""
	Class for kernel SVM
	solving
	min_alpha 1/n sum phi_{hinge}(-y_i(K @alpha)_i) + reg ||alpha||_K
	where phi_{hinge}(x) = max(1-x,0)

	reformulate as follow:
	min_alpha,ksi 1/n sum ksi_i + reg ||alpha||_K
	where
		y_i(K @alpha)_i + ksi_i - 1 >= 0
		ksi_i >= 0

	or see
	(slide 212 of class)
	min_alpha 2 alpha.T y - ||alpha||_K
	where 0<= y_i alpha_i <= 1/(2 reg n)
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
		#alpha = cp.Variable(n)
		#cons_var = cp.multiply(alpha,Y.reshape((-1,)))
		#constraint_left = np.concatenate([cons_var, cons_var])
		#constraint_right = np.concatenate([np.ones(n)/(2*self.reg*n), np.zeros(n)])
		#dual = cp.Problem(
		#	cp.Minimize(2*alpha.T@Y- alpha.T@K@alpha),
		#	[constraint_left <= constraint_right],
		#	)
		#dual.solve()
		#self.alpha = alpha.value
		y = np.diag(Y.reshape(-1))
		constraint_matrix = np.r_[y, -y]
		constraint_vector = np.concatenate([np.ones(n)/(2*self.reg*n), np.zeros(n)])
		alpha = cp.Variable(n)
		dual = cp.Problem(
			cp.Minimize(0.5 * cp.quad_form(alpha, K) - Y.reshape(-1).T @ alpha),
			[constraint_matrix @ alpha <= constraint_vector],
			)
		dual.solve()
		self.alpha = alpha.value

	def predict(self,X2):
		assert self.alpha is not None , "Fit model before predictions"
		#distance between points
		K = self.kernel(X2,self.X)
		return K@self.alpha