import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    x_new = x_new[None, :]
    Q = x_new @ W_Q # (1, d_model,) @ (d_model, d_k) -> (1, d_k)
    K = x_new @ W_K # (1, d_model,) @ (d_model, d_k) -> (1, d_k)
    V = x_new @ W_V # (1, d_model,) @ (d_model, d_v) -> (1, d_v)

    d_k = K.shape[1]
    d_v = V.shape[1]

    if cache is not None:
        K_cache, V_cache = cache[0], cache[1] # (prev_dim, d_k), (prev_dim, d_v)
    else:
        K_cache, V_cache = np.empty((0, d_k)), np.empty((0, d_v))
    K_cache = np.concatenate([K_cache, K], axis=0) # (prev_dim+1, d_k),
    V_cache = np.concatenate([V_cache, V], axis=0) # (prev_dim+1, d_k),

    scores = Q @ K_cache.T / np.sqrt(d_k) # (1, d_k) @ (d_k, prev_dim+1) -> (1, prev_dim+1)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    att_map = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # (1, prev_dim+1)
    att_scores = att_map @ V_cache # (1, prev_dim+1) @ (prev_dim+1, d_V) -> (1, d_v)
    return (att_scores[0], (K_cache, V_cache))
    