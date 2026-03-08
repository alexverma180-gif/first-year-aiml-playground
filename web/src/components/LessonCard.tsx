import { CheckCircle, Circle } from 'lucide-react';

interface LessonCardProps {
  title: string;
  description: string;
  completed: boolean;
  onClick: () => void;
  icon: React.ReactNode;
}

export function LessonCard({ title, description, completed, onClick, icon }: LessonCardProps) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left p-6 bg-white rounded-lg shadow-sm border border-slate-200 hover:shadow-md hover:border-blue-300 transition-all"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-50 rounded-lg text-blue-600">
            {icon}
          </div>
          <h3 className="text-lg font-semibold text-slate-800">{title}</h3>
        </div>
        {completed ? (
          <CheckCircle size={24} className="text-green-600 flex-shrink-0" />
        ) : (
          <Circle size={24} className="text-slate-300 flex-shrink-0" />
        )}
      </div>
      <p className="text-sm text-slate-600">{description}</p>
    </button>
  );
}
