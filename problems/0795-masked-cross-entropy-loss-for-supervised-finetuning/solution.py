import numpy as np

def masked_ce_loss(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute mean cross-entropy loss over masked (response) positions only.

    Args:
        logits: (seq_len, vocab_size) array of unnormalized scores.
        targets: (seq_len,) array of integer target token ids.
        mask: (seq_len,) boolean array; True = include in loss.

    Returns:
        Mean cross-entropy over positions where mask is True (float).
    """
    # CE = -log(p(y))
    # p(x) = exp(logits) / sum(exp(logits))
    # p(y) = exp(logits[y]) / sum(exp(logits))
    # CE = -log(exp(logits[y])/ sum(exp(logits)))
    # CE = -log(exp(logits[y])) + log(sum(exp(logits)))
    # CE = -logits[y] + log(sum(exp(logits)))
    if mask.sum() == 0:
        return 0.0
    log_y = logits[np.arange(len(targets)), targets]
    log_sum_exp_logits = np.log(np.sum(np.exp(logits), axis=-1))
    CE = -log_y + log_sum_exp_logits
    CE = CE[mask].mean()
    return CE
    