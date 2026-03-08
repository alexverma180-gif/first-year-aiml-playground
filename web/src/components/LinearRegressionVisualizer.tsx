import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { Play, RotateCcw } from 'lucide-react';
import { LinearRegressionGD, generateData } from '../utils/linearRegression';
import { supabase } from '../lib/supabase';

export function LinearRegressionVisualizer() {
  const [learningRate, setLearningRate] = useState(0.01);
  const [epochs, setEpochs] = useState(100);
  const [numSamples, setNumSamples] = useState(100);
  const [isTraining, setIsTraining] = useState(false);
  const [model, setModel] = useState<LinearRegressionGD | null>(null);
  const [lossHistory, setLossHistory] = useState<{ epoch: number; loss: number }[]>([]);
  const [dataPoints, setDataPoints] = useState<{ x: number; y: number; predicted?: number }[]>([]);

  const trainModel = async () => {
    setIsTraining(true);

    const trueWeights = [2.5];
    const trueBias = 1.0;
    const { X, y } = generateData(numSamples, 1, trueWeights, trueBias, 1.0);

    const newModel = new LinearRegressionGD(learningRate, epochs);

    setTimeout(() => {
      newModel.fit(X, y);
      setModel(newModel);
      setLossHistory(newModel.history);

      const predictions = newModel.predict(X);
      const points = X.map((x, i) => ({
        x: x[0],
        y: y[i],
        predicted: predictions[i],
      }));
      setDataPoints(points);
      setIsTraining(false);

      const finalLoss = newModel.history[newModel.history.length - 1]?.loss || 0;
      supabase
        .from('linear_regression_experiments')
        .insert({
          learning_rate: learningRate,
          epochs,
          num_samples: numSamples,
          num_features: 1,
          final_loss: finalLoss,
        })
        .then();
    }, 100);
  };

  const reset = () => {
    setModel(null);
    setLossHistory([]);
    setDataPoints([]);
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
        <h2 className="text-2xl font-semibold text-slate-800 mb-4">Linear Regression from Scratch</h2>
        <p className="text-slate-600 mb-6">
          Experiment with gradient descent parameters to see how they affect model training.
          This implementation matches the optimized version from the repository.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Learning Rate: {learningRate.toFixed(4)}
            </label>
            <input
              type="range"
              min="0.001"
              max="0.1"
              step="0.001"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              className="w-full"
              disabled={isTraining}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Epochs: {epochs}
            </label>
            <input
              type="range"
              min="10"
              max="1000"
              step="10"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              className="w-full"
              disabled={isTraining}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Samples: {numSamples}
            </label>
            <input
              type="range"
              min="50"
              max="500"
              step="50"
              value={numSamples}
              onChange={(e) => setNumSamples(parseInt(e.target.value))}
              className="w-full"
              disabled={isTraining}
            />
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={trainModel}
            disabled={isTraining}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
          >
            <Play size={18} />
            {isTraining ? 'Training...' : 'Train Model'}
          </button>
          <button
            onClick={reset}
            disabled={isTraining || !model}
            className="flex items-center gap-2 px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
          >
            <RotateCcw size={18} />
            Reset
          </button>
        </div>
      </div>

      {lossHistory.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Training Loss</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={lossHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
              <YAxis label={{ value: 'Loss (MSE)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="loss" stroke="#3b82f6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-sm text-slate-600 mt-4">
            Final Loss: <span className="font-semibold">{lossHistory[lossHistory.length - 1]?.loss.toFixed(4)}</span>
          </p>
        </div>
      )}

      {dataPoints.length > 0 && model && (
        <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Data Points & Fitted Line</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" name="X" label={{ value: 'Feature X', position: 'insideBottom', offset: -5 }} />
              <YAxis dataKey="y" name="Y" label={{ value: 'Target Y', angle: -90, position: 'insideLeft' }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              <Scatter name="Actual Data" data={dataPoints} fill="#3b82f6" />
              <Scatter name="Predictions" data={dataPoints.map(p => ({ x: p.x, y: p.predicted }))} fill="#ef4444" shape="cross" />
            </ScatterChart>
          </ResponsiveContainer>
          <div className="mt-4 p-4 bg-slate-50 rounded-lg">
            <p className="text-sm font-medium text-slate-700">Learned Parameters:</p>
            <p className="text-sm text-slate-600 mt-1">
              Weight (w): <span className="font-mono font-semibold">{model.getWeights().w[0].toFixed(4)}</span>
            </p>
            <p className="text-sm text-slate-600">
              Bias (b): <span className="font-mono font-semibold">{model.getWeights().b.toFixed(4)}</span>
            </p>
            <p className="text-xs text-slate-500 mt-2">
              True values: w = 2.5, b = 1.0
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
