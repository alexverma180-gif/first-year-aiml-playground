export class LinearRegressionGD {
  private lr: number;
  private epochs: number;
  private w: number[] | null = null;
  private b: number = 0;
  public history: { epoch: number; loss: number }[] = [];

  constructor(lr: number = 0.01, epochs: number = 1000) {
    this.lr = lr;
    this.epochs = epochs;
  }

  fit(X: number[][], y: number[]): this {
    const n = X.length;
    const d = X[0].length;
    this.w = new Array(d).fill(0);
    this.b = 0;
    this.history = [];

    const XTX = this.matmul(this.transpose(X), X);
    const XTy = this.matmul(this.transpose(X), y.map(val => [val])).map(row => row[0]);
    const X_sum = this.colSum(X);
    const y_sum = y.reduce((a, b) => a + b, 0);

    const two_over_n = 2.0 / n;
    const learning_rate_factor = this.lr * two_over_n;

    const XTX_scaled = this.scalarMul(XTX, learning_rate_factor);
    const XTy_scaled = X_sum.map((_, i) => XTy[i] * learning_rate_factor);
    const X_sum_scaled = X_sum.map(val => val * learning_rate_factor);
    const y_sum_scaled = y_sum * learning_rate_factor;
    const n_scaled = learning_rate_factor * n;

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      const step_w: number[] = this.vecAdd(
        this.matVecMul(XTX_scaled, this.w!),
        X_sum_scaled.map(val => val * this.b)
      ).map((val, i) => val - XTy_scaled[i]);

      const step_b = this.dot(X_sum_scaled, this.w!) + n_scaled * this.b - y_sum_scaled;

      this.w = this.w!.map((val, i) => val - step_w[i]);
      this.b -= step_b;

      if (epoch % Math.max(1, Math.floor(this.epochs / 100)) === 0 || epoch === this.epochs - 1) {
        const loss = this.calculateLoss(X, y);
        this.history.push({ epoch, loss });
      }
    }
    return this;
  }

  predict(X: number[][]): number[] {
    if (!this.w) {
      throw new Error('Model not trained yet');
    }
    return X.map(row => this.dot(row, this.w!) + this.b);
  }

  private calculateLoss(X: number[][], y: number[]): number {
    const predictions = this.predict(X);
    const errors = predictions.map((pred, i) => pred - y[i]);
    return errors.reduce((sum, err) => sum + err * err, 0) / X.length;
  }

  private transpose(matrix: number[][]): number[][] {
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
  }

  private matmul(a: number[][], b: number[][]): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < a.length; i++) {
      result[i] = [];
      for (let j = 0; j < b[0].length; j++) {
        let sum = 0;
        for (let k = 0; k < a[0].length; k++) {
          sum += a[i][k] * b[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  }

  private colSum(matrix: number[][]): number[] {
    const result = new Array(matrix[0].length).fill(0);
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[0].length; j++) {
        result[j] += matrix[i][j];
      }
    }
    return result;
  }

  private scalarMul(matrix: number[][], scalar: number): number[][] {
    return matrix.map(row => row.map(val => val * scalar));
  }

  private matVecMul(matrix: number[][], vec: number[]): number[] {
    return matrix.map(row => this.dot(row, vec));
  }

  private vecAdd(a: number[], b: number[]): number[] {
    return a.map((val, i) => val + b[i]);
  }

  private dot(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
  }

  getWeights(): { w: number[]; b: number } {
    if (!this.w) {
      throw new Error('Model not trained yet');
    }
    return { w: this.w, b: this.b };
  }
}

export function generateData(
  numSamples: number,
  numFeatures: number,
  trueWeights: number[],
  trueBias: number,
  noise: number = 0.5
): { X: number[][]; y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];

  for (let i = 0; i < numSamples; i++) {
    const row: number[] = [];
    for (let j = 0; j < numFeatures; j++) {
      row.push(Math.random() * 10 - 5);
    }
    X.push(row);

    let yVal = trueBias;
    for (let j = 0; j < numFeatures; j++) {
      yVal += row[j] * trueWeights[j];
    }
    yVal += (Math.random() - 0.5) * 2 * noise;
    y.push(yVal);
  }

  return { X, y };
}
