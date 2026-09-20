import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros

    # Your code here: implement gradient descent
    for _ in range(iterations):
        #loss: (1/2m)*sum((X*theta - y)^2) = (1/2m)*( (X0*theta0 - y0)^2 + (X1*theta1 - y1)^2 + ... )
        #dloss: (1/m) * ((x0*theta0 - y0)*x0 + (x1*theta1 - y1)*x1 + ...)
        #dloss: (1/m) * (X.T * X * theta - X.T * y)
        #dloss: (1/m) * X.T * (X* theta - y)
        #dloss: (1/m) * X.T * (y_hat - y)
        #dloss: (1/m) * X.T * (errors)
        y_hat = X @ theta
        errors = y_hat - y
        dloss = (1/m) * (X.T @ errors) 
        theta -= alpha*dloss
    return theta.flatten()