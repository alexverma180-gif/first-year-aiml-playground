import { useState } from 'react';
import { BookOpen, TrendingUp, Flower, Database, Hop as Home } from 'lucide-react';
import { LinearRegressionVisualizer } from './components/LinearRegressionVisualizer';
import { IrisClassifier } from './components/IrisClassifier';
import { DataCleaningTutorial } from './components/DataCleaningTutorial';
import { LessonCard } from './components/LessonCard';

type Page = 'home' | 'linear-regression' | 'iris' | 'data-cleaning';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('home');
  const [completedLessons, setCompletedLessons] = useState<Set<string>>(new Set());

  const lessons = [
    {
      id: 'linear-regression',
      title: 'Linear Regression from Scratch',
      description: 'Learn how gradient descent works by training a model with adjustable hyperparameters.',
      icon: <TrendingUp size={24} />,
    },
    {
      id: 'iris',
      title: 'Iris Species Classification',
      description: 'Build a K-Nearest Neighbors classifier to predict iris species from flower measurements.',
      icon: <Flower size={24} />,
    },
    {
      id: 'data-cleaning',
      title: 'Data Cleaning Basics',
      description: 'Master essential data preprocessing techniques including handling missing values and validation.',
      icon: <Database size={24} />,
    },
  ];

  const markLessonComplete = (lessonId: string) => {
    setCompletedLessons(prev => new Set(prev).add(lessonId));
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'linear-regression':
        return (
          <div>
            <button
              onClick={() => {
                markLessonComplete('linear-regression');
                setCurrentPage('home');
              }}
              className="mb-6 flex items-center gap-2 text-blue-600 hover:text-blue-700 font-medium"
            >
              <Home size={18} />
              Back to Home
            </button>
            <LinearRegressionVisualizer />
          </div>
        );
      case 'iris':
        return (
          <div>
            <button
              onClick={() => {
                markLessonComplete('iris');
                setCurrentPage('home');
              }}
              className="mb-6 flex items-center gap-2 text-blue-600 hover:text-blue-700 font-medium"
            >
              <Home size={18} />
              Back to Home
            </button>
            <IrisClassifier />
          </div>
        );
      case 'data-cleaning':
        return (
          <div>
            <button
              onClick={() => {
                markLessonComplete('data-cleaning');
                setCurrentPage('home');
              }}
              className="mb-6 flex items-center gap-2 text-blue-600 hover:text-blue-700 font-medium"
            >
              <Home size={18} />
              Back to Home
            </button>
            <DataCleaningTutorial />
          </div>
        );
      default:
        return (
          <div className="space-y-8">
            <div className="text-center py-12">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-600 rounded-full mb-4">
                <BookOpen size={32} className="text-white" />
              </div>
              <h1 className="text-4xl font-bold text-slate-800 mb-3">
                AIML Learning Playground
              </h1>
              <p className="text-lg text-slate-600 max-w-2xl mx-auto">
                A hands-on platform for first-year AI/ML students. Explore interactive lessons,
                train models, and master the fundamentals of machine learning.
              </p>
              <div className="mt-6 flex items-center justify-center gap-6 text-sm text-slate-500">
                <div className="flex items-center gap-2">
                  <CheckCircle size={18} className="text-green-600" />
                  <span>{completedLessons.size} / {lessons.length} Lessons Completed</span>
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-2xl font-semibold text-slate-800 mb-4">Interactive Lessons</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {lessons.map(lesson => (
                  <LessonCard
                    key={lesson.id}
                    title={lesson.title}
                    description={lesson.description}
                    completed={completedLessons.has(lesson.id)}
                    onClick={() => setCurrentPage(lesson.id as Page)}
                    icon={lesson.icon}
                  />
                ))}
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
              <h2 className="text-xl font-semibold text-slate-800 mb-4">About This Project</h2>
              <div className="space-y-3 text-slate-600">
                <p>
                  This interactive learning platform is built from the first-year-aiml-playground repository,
                  transforming static code examples into engaging, hands-on lessons.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                  <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <h3 className="font-semibold text-blue-900 mb-2">Learn by Doing</h3>
                    <p className="text-sm text-blue-800">
                      Interact with real ML algorithms and see how parameters affect model performance.
                    </p>
                  </div>
                  <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                    <h3 className="font-semibold text-green-900 mb-2">Track Progress</h3>
                    <p className="text-sm text-green-800">
                      Your experiments and progress are saved automatically as you learn.
                    </p>
                  </div>
                  <div className="p-4 bg-amber-50 rounded-lg border border-amber-200">
                    <h3 className="font-semibold text-amber-900 mb-2">Build Intuition</h3>
                    <p className="text-sm text-amber-800">
                      Visualizations help you understand complex concepts through exploration.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {renderPage()}
      </div>
    </div>
  );
}

function CheckCircle({ size, className }: { size: number; className: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" className={className}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

export default App;
