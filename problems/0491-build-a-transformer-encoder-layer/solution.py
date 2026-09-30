import numpy as np

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    d_h = Q.shape[-1]
    K_T = np.transpose(K, (0, 2, 1))
    scores = Q @ K_T / np.sqrt(d_h)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    att_map = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return att_map @ V

def layerNorm(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    norm = (X - mu) / np.sqrt(var + eps)
    return gamma * norm + beta

def FFN(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    layer1 = X @ W1 + b1
    relu = layer1
    relu = np.maximum(0, layer1)
    layer2 = relu @ W2 + b2
    return layer2

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """

    batch_size, seq_len, d_model = X.shape[0], X.shape[1], X.shape[2]
    
    W_q, W_k, W_v, W_o = weights["W_q"], weights["W_k"], weights["W_v"], weights["W_o"]
    W1, b1, W2, b2 = weights["W1"], weights["b1"], weights["W2"], weights["b2"]
    gamma1, beta1, gamma2, beta2 = weights["gamma1"], weights["beta1"], weights["gamma2"], weights["beta2"]

    Q = X @ W_q # [batch_size, seq_len, d_model]
    K = X @ W_k # [batch_size, seq_len, d_model]
    V = X @ W_v # [batch_size, seq_len, d_model]

    d_h = d_model // num_heads

    Q = Q.reshape(batch_size, seq_len, num_heads, d_h) 
    K = K.reshape(batch_size, seq_len, num_heads, d_h)
    V = V.reshape(batch_size, seq_len, num_heads, d_h)

    Q = np.transpose(Q, (2, 0, 1, 3)) # [nh, bs, sl, dh]
    K = np.transpose(K, (2, 0, 1, 3)) # [nh, bs, sl, dh]
    V = np.transpose(V, (2, 0, 1, 3)) # [nh, bs, sl, dh]

    heads = np.stack([self_attention(Q[h, :], K[h, :], V[h, :]) for h in range(num_heads)], axis=0) #[nh, bs, sl, dh]

    heads = np.transpose(heads, (1, 2, 0, 3)) # [bs, sl, nh, dh]
    heads = np.reshape(heads, (batch_size, seq_len, d_model)) # [bs, sl, dm]
    MHA_out = heads @ W_o # [bs, sl, dm]

    X = X + MHA_out # [bs, sl, dm]

    LN1 = layerNorm(X, gamma1, beta1, eps)

    FFN_out = FFN(LN1, W1, b1, W2, b2)
    
    X = LN1 + FFN_out

    LN2 = layerNorm(X, gamma2, beta2, eps)
    
    return LN2