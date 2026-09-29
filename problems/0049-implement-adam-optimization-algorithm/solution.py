import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here
    xt = x0
    mt = 0
    vt = 0
    for t in range(1, num_iterations+1):
        mt = beta1 * mt + (1 - beta1) * grad(xt)
        vt = beta2 * vt + (1 - beta2) * grad(xt)**2
        mt_hat = mt / (1 - beta1**t)
        vt_hat = vt / (1 - beta2**t)
        xt -= learning_rate * (mt_hat / (np.sqrt(vt_hat) + epsilon))
    return xt