import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	scores = np.array(scores)
	exp_scores = np.exp(scores) 
	return np.log(exp_scores / exp_scores.sum())