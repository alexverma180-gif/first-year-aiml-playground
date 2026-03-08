export interface IrisDataPoint {
  sepal_length: number;
  sepal_width: number;
  petal_length: number;
  petal_width: number;
  species: string;
}

export const irisDataset: IrisDataPoint[] = [
  { sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2, species: 'setosa' },
  { sepal_length: 4.9, sepal_width: 3.0, petal_length: 1.4, petal_width: 0.2, species: 'setosa' },
  { sepal_length: 4.7, sepal_width: 3.2, petal_length: 1.3, petal_width: 0.2, species: 'setosa' },
  { sepal_length: 4.6, sepal_width: 3.1, petal_length: 1.5, petal_width: 0.2, species: 'setosa' },
  { sepal_length: 5.0, sepal_width: 3.6, petal_length: 1.4, petal_width: 0.2, species: 'setosa' },
  { sepal_length: 5.4, sepal_width: 3.9, petal_length: 1.7, petal_width: 0.4, species: 'setosa' },
  { sepal_length: 4.6, sepal_width: 3.4, petal_length: 1.4, petal_width: 0.3, species: 'setosa' },
  { sepal_length: 5.0, sepal_width: 3.4, petal_length: 1.5, petal_width: 0.2, species: 'setosa' },
  { sepal_length: 4.9, sepal_width: 3.1, petal_length: 1.5, petal_width: 0.1, species: 'setosa' },
  { sepal_length: 5.4, sepal_width: 3.7, petal_length: 1.5, petal_width: 0.2, species: 'setosa' },
  { sepal_length: 6.4, sepal_width: 3.2, petal_length: 4.5, petal_width: 1.5, species: 'versicolor' },
  { sepal_length: 6.9, sepal_width: 3.1, petal_length: 4.9, petal_width: 1.5, species: 'versicolor' },
  { sepal_length: 5.5, sepal_width: 2.3, petal_length: 4.0, petal_width: 1.3, species: 'versicolor' },
  { sepal_length: 6.5, sepal_width: 2.8, petal_length: 4.6, petal_width: 1.5, species: 'versicolor' },
  { sepal_length: 5.7, sepal_width: 2.8, petal_length: 4.5, petal_width: 1.3, species: 'versicolor' },
  { sepal_length: 6.0, sepal_width: 2.2, petal_length: 4.0, petal_width: 1.0, species: 'versicolor' },
  { sepal_length: 6.1, sepal_width: 2.9, petal_length: 4.7, petal_width: 1.4, species: 'versicolor' },
  { sepal_length: 5.6, sepal_width: 2.9, petal_length: 3.6, petal_width: 1.3, species: 'versicolor' },
  { sepal_length: 6.7, sepal_width: 3.1, petal_length: 4.4, petal_width: 1.4, species: 'versicolor' },
  { sepal_length: 6.3, sepal_width: 3.3, petal_length: 6.0, petal_width: 2.5, species: 'virginica' },
  { sepal_length: 5.8, sepal_width: 2.7, petal_length: 5.1, petal_width: 1.9, species: 'virginica' },
  { sepal_length: 7.1, sepal_width: 3.0, petal_length: 5.9, petal_width: 2.1, species: 'virginica' },
  { sepal_length: 6.3, sepal_width: 2.9, petal_length: 5.6, petal_width: 1.8, species: 'virginica' },
  { sepal_length: 6.5, sepal_width: 3.0, petal_length: 5.8, petal_width: 2.2, species: 'virginica' },
  { sepal_length: 7.6, sepal_width: 3.0, petal_length: 6.6, petal_width: 2.1, species: 'virginica' },
  { sepal_length: 6.8, sepal_width: 3.2, petal_length: 5.9, petal_width: 2.3, species: 'virginica' },
  { sepal_length: 6.7, sepal_width: 3.3, petal_length: 5.7, petal_width: 2.5, species: 'virginica' },
  { sepal_length: 6.7, sepal_width: 3.0, petal_length: 5.2, petal_width: 2.3, species: 'virginica' },
];

export class KNNClassifier {
  private k: number;
  private X_train: number[][] = [];
  private y_train: string[] = [];

  constructor(k: number = 5) {
    this.k = k;
  }

  fit(X: number[][], y: string[]): void {
    this.X_train = X;
    this.y_train = y;
  }

  predict(X: number[][]): string[] {
    return X.map(point => this.predictSingle(point));
  }

  predictSingle(point: number[]): string {
    const distances = this.X_train.map((trainPoint, idx) => ({
      distance: this.euclideanDistance(point, trainPoint),
      label: this.y_train[idx],
    }));

    distances.sort((a, b) => a.distance - b.distance);

    const kNearest = distances.slice(0, this.k);
    const votes: { [key: string]: number } = {};

    kNearest.forEach(({ label }) => {
      votes[label] = (votes[label] || 0) + 1;
    });

    return Object.entries(votes).reduce((a, b) => (b[1] > a[1] ? b : a))[0];
  }

  private euclideanDistance(a: number[], b: number[]): number {
    return Math.sqrt(
      a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0)
    );
  }
}

export function prepareIrisData(
  data: IrisDataPoint[],
  testSize: number = 0.2
): {
  X_train: number[][];
  X_test: number[][];
  y_train: string[];
  y_test: string[];
} {
  const shuffled = [...data].sort(() => Math.random() - 0.5);
  const splitIndex = Math.floor(shuffled.length * (1 - testSize));

  const train = shuffled.slice(0, splitIndex);
  const test = shuffled.slice(splitIndex);

  const X_train = train.map(d => [
    d.sepal_length,
    d.sepal_width,
    d.petal_length,
    d.petal_width,
  ]);
  const y_train = train.map(d => d.species);

  const X_test = test.map(d => [
    d.sepal_length,
    d.sepal_width,
    d.petal_length,
    d.petal_width,
  ]);
  const y_test = test.map(d => d.species);

  return { X_train, X_test, y_train, y_test };
}

export function calculateAccuracy(predictions: string[], actual: string[]): number {
  const correct = predictions.filter((pred, i) => pred === actual[i]).length;
  return correct / predictions.length;
}
