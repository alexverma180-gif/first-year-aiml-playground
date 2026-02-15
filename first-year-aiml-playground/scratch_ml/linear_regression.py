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

        # Precompute values for analytical gradient to reduce complexity per iteration from O(N*D) to O(D^2).
        # We use an "Augmented Matrix" approach to combine weight and bias updates into a single operation.
        # Instead of explicitly creating an augmented X matrix (which copies data), we construct
        # the augmented covariance matrices from precomputed terms to save memory and time.
        XTX = X.T @ X
        XTy = X.T @ y
        X_sum = X.sum(axis=0)
        y_sum = y.sum()

        # Construct augmented XTX_aug = [[XTX, X_sum], [X_sum.T, n]]
        XTX_aug = np.empty((d + 1, d + 1))
        XTX_aug[:d, :d] = XTX
        XTX_aug[:d, d] = X_sum
        XTX_aug[d, :d] = X_sum
        XTX_aug[d, d] = n

        # Construct augmented XTy_aug = [XTy, y_sum]
        XTy_aug = np.empty(d + 1)
        XTy_aug[:d] = XTy
        XTy_aug[d] = y_sum

        # Pre-scale terms by the learning rate factor outside the loop to minimize operations inside.
        learning_rate_factor = self.lr * (2.0 / n)
        XTX_scaled = learning_rate_factor * XTX_aug
        XTy_scaled = learning_rate_factor * XTy_aug

        theta = np.zeros(d + 1)
        for _ in range(self.epochs):
            # Combined analytical gradient calculation:
            # theta = theta - factor * (XTX_aug @ theta - XTy_aug)
            # theta = theta - (XTX_scaled @ theta - XTy_scaled)
            theta -= (XTX_scaled @ theta - XTy_scaled)

        self.w = theta[:d]
        self.b = theta[d]
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
