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

        two_over_n = 2.0 / n
        learning_rate_factor = self.lr * two_over_n

        # Precompute constants to reduce complexity from O(n*d) to O(d^2) per iteration
        XtX = X.T @ X
        Xty = X.T @ y
        col_sums = X.sum(axis=0)
        sum_y = y.sum()

        for _ in range(self.epochs):
            # Gradient wrt w: (2/n) * (XtX @ w + b * col_sums - Xty)
            # Gradient wrt b: (2/n) * (col_sums @ w + n * b - sum_y)
            grad_w = XtX @ self.w
            grad_w += self.b * col_sums
            grad_w -= Xty

            grad_b = col_sums @ self.w
            grad_b += n * self.b
            grad_b -= sum_y

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
