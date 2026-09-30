import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    return X @ W_q, X @ W_k, X @ W_v

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    d_k = K.shape[1]
    scores = (Q @ K.T) / torch.sqrt(torch.tensor(d_k))
    scores = scores - scores.max(dim=1, keepdim=True).values
    att_map = F.softmax(scores, dim=1)
    return att_map @ V

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_model = Q.shape[0], Q.shape[1]
    d_k = d_model // n_heads
    Q = Q.view(seq_len, n_heads, d_k).transpose(0, 1) # [n_heads, seq_heads, d_k]
    K = K.view(seq_len, n_heads, d_k).transpose(0, 1) # [n_heads, seq_heads, d_k]
    V = V.view(seq_len, n_heads, d_k).transpose(0, 1) # [n_heads, seq_heads, d_k]
    
    heads = torch.stack([self_attention(Q[h], K[h], V[h]) for h in range(n_heads)]) 

    return heads.transpose(0, 1).reshape(seq_len, d_model)
