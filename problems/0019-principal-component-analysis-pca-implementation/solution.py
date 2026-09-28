import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    stand_data = (data - data.mean(axis=0)) / data.std(axis=0)
    cov = np.cov(stand_data, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    pcs = eigenvectors[:, order[:k]]
    for i in range(pcs.shape[1]):
        vec = pcs[:, i]
        nonzero = np.where(np.abs(vec) > 1e-10)[0]
        if vec[nonzero[0]] < 0:
            pcs[:, i] = -vec
    return np.round(pcs, 4)
