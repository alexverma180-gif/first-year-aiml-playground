import numpy as np
from scratch_ml.linear_regression import LinearRegressionGD

def test_learns_simple_line():
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, size=(120, 1))
    y = 3*X[:,0] + 2 + rng.normal(0, 0.2, size=120)
    model = LinearRegressionGD(lr=0.05, epochs=2000).fit(X, y)
    preds = model.predict([[0],[1],[2]]).ravel()
    assert abs(preds[0] - 2) < 0.6
    assert abs(preds[1] - 5) < 0.6
    assert abs(preds[2] - 8) < 0.6
