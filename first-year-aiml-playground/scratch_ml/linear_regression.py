import numpy as np

class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        # Precompute values for analytical gradient to reduce complexity per iteration from O(N*D) to O(D^2).
        # This optimization is highly effective when N >> D and the number of epochs is large.
        # Note: Precomputing XTX has an initial cost of O(N*D^2) and requires O(D^2) additional space.
        # For very high-dimensional data (large D), this might lead to increased memory usage.
        XTX = X.T @ X
        XTy = X.T @ y
        X_sum = X.sum(axis=0)
        y_sum = y.sum()

        two_over_n = 2.0 / n
        learning_rate_factor = self.lr * two_over_n

        # Use an Augmented Matrix approach to combine w and b into a single weight vector.
        # This reduces the training loop to a single matrix-vector multiplication,
        # further minimizing Python loop overhead and better utilizing BLAS.
        # Augmented covariance matrix: [[XTX, X_sum], [X_sum.T, n]]
        M = np.empty((d + 1, d + 1))
        M[:d, :d] = XTX
        M[:d, d] = X_sum
        M[d, :d] = X_sum
        M[d, d] = n

        # Augmented XTy vector: [XTy, y_sum]
        V = np.empty(d + 1)
        V[:d] = XTy
        V[d] = y_sum

        # Pre-scale by learning rate factor
        M_scaled = learning_rate_factor * M
        V_scaled = learning_rate_factor * V

        # Augmented weight vector [w_1, ..., w_d, b]
        W = np.zeros(d + 1)
        # Initialize from existing self.w and self.b (already set to zeros at start of fit)
        W[:d] = self.w
        W[d] = self.b

        for _ in range(self.epochs):
            # Combined gradient step: W = W - (M_scaled @ W - V_scaled)
            W -= (M_scaled @ W - V_scaled)

        self.w = W[:d]
        self.b = float(W[d])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

if __name__ == "__main__":
    # tiny demo
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, size=(100, 1))
    y = 3*X[:,0] + 2 + rng.normal(0, 0.2, size=100)
    model = LinearRegressionGD(lr=0.05, epochs=2000).fit(X, y)
    preds = model.predict([[0],[1],[2]]).ravel()
    print("Predictions for x=[0,1,2]:", preds)
