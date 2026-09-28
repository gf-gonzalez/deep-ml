import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	features = np.array(features)
	weights = np.array(weights)
	labels = np.array(labels)
	logits = features @ weights + bias
	probabilities = (1 / (1 + np.exp(-logits)))
	n = len(labels)
	mse = (1/n)*((probabilities-labels)**2).sum()
	return np.round(probabilities, 4), np.round(mse, 4)