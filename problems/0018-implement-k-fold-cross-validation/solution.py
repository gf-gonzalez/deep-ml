import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    samples = np.arange(n_samples).tolist()
    if shuffle == True:
        np.random.shuffle(samples)
    size = n_samples // k
    folds = [samples[i*size:(i+1)*size] if (i < (k-1)) and (n_samples % k == 0) else samples[i*size:] for i in range(k-1, -1, -1)][::-1]
    fdict = {i:f for i,f in enumerate(folds)}
    kfolds_list = []
    for i in range(k):
        test_indices = fdict[i]
        train_indices = []
        for f, idxs in fdict.items():
            if f != i:
                train_indices += idxs
        kfolds_list.append((train_indices, test_indices))
    return kfolds_list

