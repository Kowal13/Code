import numpy as np

def calculate_L(beta_hat: np.ndarray, beta: np.ndarray, A: np.ndarray) -> float:
    diff = beta_hat - beta
    L = np.dot(np.dot(diff.T, A), diff)
    return L


A = np.array([[2, 1, 0],
              [1, 4, 2],
              [0, 2, 6]])  # Example symmetric matrix

Beta_hat = np.array([1, 2, 3])  # Example Beta_hat vector
Beta = np.array([4, 5, 6])  # Example Beta vector

L_value = calculate_L(Beta_hat, Beta, A)
print("L =", L_value)


def least_squares_hat(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Add a column of ones to the independent variable matrix for the intercept term
    X = np.vstack((np.ones(len(x)), x)).T

    # Compute the least squares estimator
    beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    return beta_hat

# Example usage
x = np.array([1, 2, 3, 4, 5])  # Independent variable
y = np.array([2, 3.9, 6.1, 8.2, 10.3])  # Dependent variable

beta_hat = least_squares_hat(x, y)
print("Intercept:", beta_hat[0])
print("Slope:", beta_hat[1])

def replace_negative_entries(A: np.ndarray) -> np.ndarray:
    return np.where(A < 0, 0, A)

# Example usage
A = np.array([[1, -2, 3],
              [-4, 5, -6],
              [7, 8, -9]])

result = replace_negative_entries(A)
print(result)

def estimate_h_monte_carlo(X: np.ndarray, A: np.ndarray, B: np.ndarray, sigma=1, rho=1, num_iterations=1000) -> float:
    h_estimates = []
    
    for _ in range(num_iterations):
        # Generate a random value of h
        h = np.random.uniform(0.1, 10.0)  # Adjust the range as needed
        
        # Calculate f(h^(-1) * A^(1/2) * B^(1/2) - B)
        f_value = replace_negative_entries(h ** (-1) * np.sqrt(A) @ np.sqrt(B) - B)
        
        # Calculate (X'X)^(-1)
        inverse_XTX = np.linalg.inv(X.T @ X)
        
        # Calculate tr((X'X)^(-1) * f(h^(-1) * A^(1/2) * B^(1/2) - B))
        trace_value = np.trace(inverse_XTX @ f_value)
        
        # Calculate the value of h_est if trace_value is non-zero
        if trace_value != 0:
            h_est = (rho / (sigma ** 2)) / trace_value
            h_estimates.append(h_est)
    
    # Return the average of the estimated h values
    return np.mean(h_estimates)

def replace_negative_entries(A: np.ndarray) -> np.ndarray:
    return np.where(A < 0, 0, A)

# Example usage
n = 3  # Size of X matrix
p = 2  # Size of A and B matrices

X = np.random.rand(n, p)  # Example X matrix
A = np.random.rand(p, p)  # Example A matrix
B = np.random.rand(p, p)  # Example B matrix

sigma = 1.0  # Example value of sigma
rho = 2.0  # Example value of rho
num_iterations = 1000  # Number of Monte Carlo iterations

result = estimate_h_monte_carlo(X, A, B, sigma, rho, num_iterations)
print("Estimated value of h:", result)

def compute_beta_ml_hat(A: np.ndarray, B: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    p = A.shape[0]  # Dimension of A and B
    n = Y.shape[0]  # Size of Y vector
    I = np.eye(p)  # Identity matrix

    # Calculate B^(1/2) and A^(-1/2)
    sqrt_B = np.sqrt(B)
    inv_sqrt_A = np.linalg.inv(np.sqrt(A))

    # Calculate the term inside f()
    term = I - estimate_h_monte_carlo(X, A, B, sigma=1, rho=1, num_iterations=1000) * sqrt_B @ inv_sqrt_A

    # Calculate f(I - h * B^(1/2) * A^(-1/2)) and g(Y)
    f_value = replace_negative_entries(term)
    g_value = least_squares_hat(X, Y)

    # Calculate beta_ml_hat
    beta_ml_hat = np.dot(f_value, g_value)

    return beta_ml_hat

def replace_negative_entries(A: np.ndarray) -> np.ndarray:
    return np.where(A < 0, 0, A)


# Example usage
p = 3  # Dimension of A and B
n = 5  # Size of Y vector

A = np.random.rand(p, p)  # Example A matrix
B = np.random.rand(p, p)  # Example B matrix
Y = np.random.rand(n, p)  # Example X vector
Y = np.random.rand(n, 1)  # Example Y vector
h = 0.5  # Example value of h

beta_ml_hat = compute_beta_ml_hat(A, B, X, Y)
print("beta_ml_hat:")
print(beta_ml_hat)