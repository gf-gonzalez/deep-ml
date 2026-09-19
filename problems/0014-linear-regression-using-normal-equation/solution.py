import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	# W = ((X^T X)^-1 X^T) y
	# y (n x 1)
	X = np.array(X) # (n x d+1)
	X_t = np.transpose(X) # (d+1 x n)
	X2_inv = np.linalg.inv(np.matmul(X_t, X)) # (d+1 x d+1)
	A = np.matmul(X2_inv, X_t) # (d+1 x n)
	theta = np.matmul(A, y) # (d+1 x 1)
	theta = np.round(theta, 4).tolist()
	return theta