import { useState } from 'react';
import { CheckCircle, XCircle, AlertCircle } from 'lucide-react';

interface DataRow {
  id: number;
  name: string;
  age: number | null;
  email: string;
  salary: number | null;
}

const rawData: DataRow[] = [
  { id: 1, name: 'Alice Smith', age: 28, email: 'alice@example.com', salary: 65000 },
  { id: 2, name: 'Bob Jones', age: null, email: 'bob@example', salary: 72000 },
  { id: 3, name: '', age: 35, email: 'carol@example.com', salary: null },
  { id: 4, name: 'David Lee', age: 42, email: 'david@example.com', salary: 85000 },
  { id: 5, name: 'Eve Brown', age: null, email: 'eve@example.com', salary: 68000 },
  { id: 6, name: 'Frank Wilson', age: 29, email: 'frank@invalid', salary: 71000 },
];

export function DataCleaningTutorial() {
  const [step, setStep] = useState(0);
  const [cleanedData, setCleanedData] = useState<DataRow[]>(rawData);
  const [issuesFound, setIssuesFound] = useState<string[]>([]);

  const steps = [
    {
      title: 'Identify Missing Values',
      description: 'Find rows with null or empty values that need attention.',
      action: () => {
        const issues: string[] = [];
        rawData.forEach(row => {
          if (row.age === null) issues.push(`Row ${row.id}: Missing age`);
          if (row.salary === null) issues.push(`Row ${row.id}: Missing salary`);
          if (!row.name) issues.push(`Row ${row.id}: Missing name`);
        });
        setIssuesFound(issues);
      }
    },
    {
      title: 'Handle Missing Numeric Values',
      description: 'Fill missing age and salary with mean values.',
      action: () => {
        const validAges = rawData.filter(r => r.age !== null).map(r => r.age!);
        const meanAge = Math.round(validAges.reduce((a, b) => a + b, 0) / validAges.length);

        const validSalaries = rawData.filter(r => r.salary !== null).map(r => r.salary!);
        const meanSalary = Math.round(validSalaries.reduce((a, b) => a + b, 0) / validSalaries.length);

        const cleaned = rawData.map(row => ({
          ...row,
          age: row.age ?? meanAge,
          salary: row.salary ?? meanSalary,
        }));
        setCleanedData(cleaned);
      }
    },
    {
      title: 'Validate Email Format',
      description: 'Check for invalid email addresses and flag them.',
      action: () => {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        const issues: string[] = [];
        cleanedData.forEach(row => {
          if (!emailRegex.test(row.email)) {
            issues.push(`Row ${row.id}: Invalid email format`);
          }
        });
        setIssuesFound(issues);
      }
    },
    {
      title: 'Remove Invalid Rows',
      description: 'Filter out rows with invalid data that cannot be fixed.',
      action: () => {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        const cleaned = cleanedData.filter(row => row.name && emailRegex.test(row.email));
        setCleanedData(cleaned);
        setIssuesFound([`Removed ${rawData.length - cleaned.length} invalid rows`]);
      }
    }
  ];

  const handleStepAction = () => {
    steps[step].action();
    if (step < steps.length - 1) {
      setTimeout(() => setStep(step + 1), 1500);
    }
  };

  const reset = () => {
    setStep(0);
    setCleanedData(rawData);
    setIssuesFound([]);
  };

  const getRowStatus = (row: DataRow): 'clean' | 'warning' | 'error' => {
    if (step === 0) {
      if (!row.name || row.age === null || row.salary === null) return 'error';
    }
    if (step >= 2) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(row.email)) return 'warning';
    }
    return 'clean';
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
        <h2 className="text-2xl font-semibold text-slate-800 mb-4">Data Cleaning Basics</h2>
        <p className="text-slate-600 mb-6">
          Learn the fundamentals of data preprocessing with this interactive tutorial.
          Follow each step to clean a sample dataset.
        </p>

        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-slate-800">
              Step {step + 1}: {steps[step].title}
            </h3>
            <span className="text-sm text-slate-500">
              {step + 1} of {steps.length}
            </span>
          </div>
          <p className="text-slate-600 mb-4">{steps[step].description}</p>

          <div className="flex gap-3">
            <button
              onClick={handleStepAction}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              {step === 0 ? 'Start Analysis' : 'Apply Step'}
            </button>
            <button
              onClick={reset}
              className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition-colors"
            >
              Reset
            </button>
          </div>
        </div>

        {issuesFound.length > 0 && (
          <div className="mb-6 p-4 bg-amber-50 rounded-lg border border-amber-200">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle size={18} className="text-amber-700" />
              <h4 className="font-semibold text-amber-900">Issues Found:</h4>
            </div>
            <ul className="list-disc list-inside text-sm text-amber-800 space-y-1">
              {issuesFound.map((issue, idx) => (
                <li key={idx}>{issue}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="bg-slate-100">
                <th className="px-4 py-2 text-left text-sm font-semibold text-slate-700 border">ID</th>
                <th className="px-4 py-2 text-left text-sm font-semibold text-slate-700 border">Name</th>
                <th className="px-4 py-2 text-left text-sm font-semibold text-slate-700 border">Age</th>
                <th className="px-4 py-2 text-left text-sm font-semibold text-slate-700 border">Email</th>
                <th className="px-4 py-2 text-left text-sm font-semibold text-slate-700 border">Salary</th>
                <th className="px-4 py-2 text-center text-sm font-semibold text-slate-700 border">Status</th>
              </tr>
            </thead>
            <tbody>
              {cleanedData.map((row) => {
                const status = getRowStatus(row);
                return (
                  <tr key={row.id} className={
                    status === 'error' ? 'bg-red-50' :
                    status === 'warning' ? 'bg-amber-50' :
                    'bg-white'
                  }>
                    <td className="px-4 py-2 text-sm text-slate-700 border">{row.id}</td>
                    <td className="px-4 py-2 text-sm text-slate-700 border">
                      {row.name || <span className="text-red-500 italic">empty</span>}
                    </td>
                    <td className="px-4 py-2 text-sm text-slate-700 border">
                      {row.age ?? <span className="text-red-500 italic">null</span>}
                    </td>
                    <td className="px-4 py-2 text-sm text-slate-700 border">{row.email}</td>
                    <td className="px-4 py-2 text-sm text-slate-700 border">
                      {row.salary ? `$${row.salary.toLocaleString()}` : <span className="text-red-500 italic">null</span>}
                    </td>
                    <td className="px-4 py-2 text-center border">
                      {status === 'clean' && <CheckCircle size={18} className="text-green-600 mx-auto" />}
                      {status === 'warning' && <AlertCircle size={18} className="text-amber-600 mx-auto" />}
                      {status === 'error' && <XCircle size={18} className="text-red-600 mx-auto" />}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Key Concepts</h3>
        <div className="space-y-3">
          <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
            <h4 className="font-semibold text-blue-900 mb-1">Missing Values</h4>
            <p className="text-sm text-blue-800">
              Identify and handle null or empty values using strategies like mean imputation or removal.
            </p>
          </div>
          <div className="p-3 bg-green-50 rounded-lg border border-green-200">
            <h4 className="font-semibold text-green-900 mb-1">Data Validation</h4>
            <p className="text-sm text-green-800">
              Check data against expected formats and constraints to ensure quality.
            </p>
          </div>
          <div className="p-3 bg-amber-50 rounded-lg border border-amber-200">
            <h4 className="font-semibold text-amber-900 mb-1">Outlier Detection</h4>
            <p className="text-sm text-amber-800">
              Identify and handle extreme values that may skew your analysis.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
