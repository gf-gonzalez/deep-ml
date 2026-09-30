import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	prev_att_map = np.dot(Q, K.T) / np.sqrt(K.shape[1])
	masked_prev_att_map = prev_att_map + mask 
	masked_prev_att_map = masked_prev_att_map - np.max(masked_prev_att_map, axis=1, keepdims=True)
	exp_vals = np.exp(masked_prev_att_map)
	att_map = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
	att_scores = np.dot(att_map, V)
	return att_scores