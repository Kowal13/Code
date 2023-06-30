import numpy as np

def compute_beta_ml_hat(A: np.ndarray, B: np.ndarray, X: np.ndarray, Y: np.ndarray, h: float) -> float:
    p = A.shape[0]  # Dimension of A and B
    n = Y.shape[0]  # Size of Y vector
    I = np.eye(p)  # Identity matrix

    # Calculate B^(1/2) and A^(-1/2)
    sqrt_B = np.sqrt(B)
    inv_sqrt_A = np.linalg.inv(np.sqrt(A))

    # Calculate the term inside f()
    term = I - h * sqrt_B @ inv_sqrt_A

    # Calculate f(I - h * B^(1/2) * A^(-1/2)) and g(Y)
    f_value = replace_negative_entries(term)
    g_value = least_squares_estimator(X, Y)

    # Calculate beta_ml_hat
    beta_ml_hat = np.trace(f_value) * np.trace(g_value)

    return beta_ml_hat

def replace_negative_entries(A: np.ndarray) -> np.ndarray:
    return np.where(A < 0, 0, A)

def least_squares_estimator(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_transpose = np.transpose(X)
    XTX_inv = np.linalg.inv(X_transpose @ X)
    beta_hat = XTX_inv @ X_transpose @ Y
    return beta_hat

# Example usage
p = 3  # Dimension of A and B
n = 5  # Size of Y vector

A = np.random.rand(p, p)  # Example A matrix
B = np.random.rand(p, p)  # Example B matrix
X = np.random.rand(n, p)  # Example X matrix
Y = np.random.rand(n, 1)  # Example Y vector
h = 0.5  # Example value of h

beta_ml_hat = compute_beta_ml_hat(A, B, X, Y, h)
print("beta_ml_hat:")
print(beta_ml_hat)
