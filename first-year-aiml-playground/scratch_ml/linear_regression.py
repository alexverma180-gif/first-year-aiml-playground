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

        # Further optimize by pre-scaling terms by the learning rate factor outside the loop.
        # This reduces the number of scalar-vector multiplications inside the loop.
        XTX_scaled = learning_rate_factor * XTX
        XTy_scaled = learning_rate_factor * XTy
        X_sum_scaled = learning_rate_factor * X_sum
        y_sum_scaled = learning_rate_factor * y_sum
        # For the bias update, we also pre-scale the 'n' and 'b' terms
        n_scaled = learning_rate_factor * n

        for _ in range(self.epochs):
            # Analytical gradient calculation with pre-scaled terms:
            # Grad w * factor = (learning_rate_factor * XTX) @ w + b * (learning_rate_factor * X_sum) - (learning_rate_factor * XTy)
            # Grad b * factor = (learning_rate_factor * X_sum) @ w + b * (learning_rate_factor * n) - (learning_rate_factor * y_sum)
            step_w = XTX_scaled @ self.w + self.b * X_sum_scaled - XTy_scaled
            step_b = X_sum_scaled @ self.w + n_scaled * self.b - y_sum_scaled

            self.w -= step_w
            self.b -= step_b
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
