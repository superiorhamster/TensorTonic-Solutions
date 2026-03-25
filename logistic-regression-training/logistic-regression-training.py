import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 1
    for step in range(steps):
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # 2. Tính toán Gradient (Backward pass)
        diffW = np.dot(X.T, (p - y)) / n_samples
        diffB = np.sum(p - y) / n_samples
        
        # 3. Cập nhật trọng số và bias
        w = w - lr * diffW
        b = b - lr * diffB

    return (w,b)