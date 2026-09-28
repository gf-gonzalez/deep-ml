import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	ones = np.ones_like(data)
	mean = np.mean(data, axis=0) * ones
	std = np.std(data, axis=0) * ones
	standardized_data = (data - mean) / std

	min_f = np.min(data, axis=0) * ones
	max_f = np.max(data, axis=0) * ones
	normalized_data = (data - min_f) / (max_f - min_f)

	return standardized_data, normalized_data