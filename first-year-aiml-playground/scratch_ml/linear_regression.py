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

        for _ in range(self.epochs):
            # Analytical gradient calculation using precomputed terms:
            # Grad w = (2/n) * (X.T @ (X @ w + b - y)) = (2/n) * (XTX @ w + b * X_sum - XTy)
            # Grad b = (2/n) * sum(X @ w + b - y) = (2/n) * (X_sum @ w + n * b - y_sum)
            grad_w = XTX @ self.w + self.b * X_sum - XTy
            grad_b = X_sum @ self.w + n * self.b - y_sum

            self.w -= learning_rate_factor * grad_w
            self.b -= learning_rate_factor * grad_b
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
