import numpy as np

def square_dist(X1,X2):
	"""
	Computes euclidean square distance matrix
	M between matricies X1 and X2:

	M[i][j] = (X1[i] -X2[j])^T(X1[i] -X2[j])
	"""

	X1_ = X1[:,None,:]
	X2_ = X2[None,:,:]

	M = np.sum((X1_ - X2_)**2,axis=2)
	return M 

def linear_kernel(X1,X2=None):
	if X2 is None:
		return X1 @ X1.T
	else:
		return X1 @ X2.T


def polynomial_kernel(X1,X2=None,deg=2):
	if X2 is None:
		return (X1@X1.T)**deg
	else:
		return (X1 @ X2.T)**deg 

#alternative to square_dist for speed
from scipy.spatial import distance_matrix

def exp_kernel(X1,X2=None,l=1):
	if X2 is None:
		M = distance_matrix(X1,X1)**2
	else:
		M = distance_matrix(X1,X2)**2
	return np.exp(-M/ (2*l))


