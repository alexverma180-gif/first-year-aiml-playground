import { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Brain, RotateCcw } from 'lucide-react';
import { KNNClassifier, prepareIrisData, calculateAccuracy, irisDataset } from '../utils/irisClassifier';
import { supabase } from '../lib/supabase';

const speciesColors: { [key: string]: string } = {
  setosa: '#3b82f6',
  versicolor: '#10b981',
  virginica: '#f59e0b',
};

export function IrisClassifier() {
  const [k, setK] = useState(5);
  const [testSize, setTestSize] = useState(0.2);
  const [model, setModel] = useState<KNNClassifier | null>(null);
  const [accuracy, setAccuracy] = useState<number | null>(null);
  const [isTrained, setIsTrained] = useState(false);

  const [sepalLength, setSepalLength] = useState(5.1);
  const [sepalWidth, setSepalWidth] = useState(3.5);
  const [petalLength, setPetalLength] = useState(1.4);
  const [petalWidth, setPetalWidth] = useState(0.2);
  const [prediction, setPrediction] = useState<string | null>(null);

  const trainModel = () => {
    const { X_train, X_test, y_train, y_test } = prepareIrisData(irisDataset, testSize);

    const knn = new KNNClassifier(k);
    knn.fit(X_train, y_train);

    const predictions = knn.predict(X_test);
    const acc = calculateAccuracy(predictions, y_test);

    setModel(knn);
    setAccuracy(acc);
    setIsTrained(true);
  };

  const makePrediction = async () => {
    if (!model) return;

    const features = [[sepalLength, sepalWidth, petalLength, petalWidth]];
    const pred = model.predict(features)[0];
    setPrediction(pred);

    await supabase.from('iris_predictions').insert({
      sepal_length: sepalLength,
      sepal_width: sepalWidth,
      petal_length: petalLength,
      petal_width: petalWidth,
      predicted_species: pred,
    });
  };

  const reset = () => {
    setModel(null);
    setAccuracy(null);
    setIsTrained(false);
    setPrediction(null);
  };

  const chartData = irisDataset.map(d => ({
    x: d.petal_length,
    y: d.petal_width,
    species: d.species,
  }));

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
        <h2 className="text-2xl font-semibold text-slate-800 mb-4">Iris Species Classifier</h2>
        <p className="text-slate-600 mb-6">
          Train a K-Nearest Neighbors classifier to predict iris species based on flower measurements.
          Adjust the K value to see how it affects model accuracy.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              K (Neighbors): {k}
            </label>
            <input
              type="range"
              min="1"
              max="15"
              step="1"
              value={k}
              onChange={(e) => setK(parseInt(e.target.value))}
              className="w-full"
              disabled={isTrained}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Test Size: {(testSize * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.1"
              max="0.5"
              step="0.05"
              value={testSize}
              onChange={(e) => setTestSize(parseFloat(e.target.value))}
              className="w-full"
              disabled={isTrained}
            />
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={trainModel}
            disabled={isTrained}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
          >
            <Brain size={18} />
            Train Model
          </button>
          <button
            onClick={reset}
            disabled={!isTrained}
            className="flex items-center gap-2 px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
          >
            <RotateCcw size={18} />
            Reset
          </button>
        </div>

        {accuracy !== null && (
          <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
            <p className="text-sm font-medium text-green-900">
              Model Accuracy: <span className="text-lg font-bold">{(accuracy * 100).toFixed(1)}%</span>
            </p>
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Dataset Visualization</h3>
        <ResponsiveContainer width="100%" height={350}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" name="Petal Length" label={{ value: 'Petal Length (cm)', position: 'insideBottom', offset: -5 }} />
            <YAxis dataKey="y" name="Petal Width" label={{ value: 'Petal Width (cm)', angle: -90, position: 'insideLeft' }} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend />
            <Scatter name="Setosa" data={chartData.filter(d => d.species === 'setosa')} fill={speciesColors.setosa} />
            <Scatter name="Versicolor" data={chartData.filter(d => d.species === 'versicolor')} fill={speciesColors.versicolor} />
            <Scatter name="Virginica" data={chartData.filter(d => d.species === 'virginica')} fill={speciesColors.virginica} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {isTrained && (
        <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Make a Prediction</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Sepal Length: {sepalLength.toFixed(1)} cm
              </label>
              <input
                type="range"
                min="4.0"
                max="8.5"
                step="0.1"
                value={sepalLength}
                onChange={(e) => setSepalLength(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Sepal Width: {sepalWidth.toFixed(1)} cm
              </label>
              <input
                type="range"
                min="2.0"
                max="4.5"
                step="0.1"
                value={sepalWidth}
                onChange={(e) => setSepalWidth(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Petal Length: {petalLength.toFixed(1)} cm
              </label>
              <input
                type="range"
                min="1.0"
                max="7.0"
                step="0.1"
                value={petalLength}
                onChange={(e) => setPetalLength(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Petal Width: {petalWidth.toFixed(1)} cm
              </label>
              <input
                type="range"
                min="0.1"
                max="2.5"
                step="0.1"
                value={petalWidth}
                onChange={(e) => setPetalWidth(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>

          <button
            onClick={makePrediction}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Predict Species
          </button>

          {prediction && (
            <div className="mt-4 p-4 rounded-lg border-2" style={{
              backgroundColor: `${speciesColors[prediction]}15`,
              borderColor: speciesColors[prediction]
            }}>
              <p className="text-sm font-medium" style={{ color: speciesColors[prediction] }}>
                Predicted Species: <span className="text-xl font-bold capitalize">{prediction}</span>
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
