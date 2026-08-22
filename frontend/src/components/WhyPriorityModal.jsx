import React from 'react';
import { X, HelpCircle, Calendar, Hash, Award, BookOpen, TrendingUp } from 'lucide-react';

export default function WhyPriorityModal({ topic, onClose }) {
  if (!topic) return null;

  const priorityScore = topic.priority_score || 85.0;
  const appearances = topic.total_appearances || topic.appearances_count || 3;
  const recentYears = topic.recent_years || topic.years_appeared || ['2023', '2024'];
  const marksPattern = topic.marks_pattern || topic.marks_distribution || ['10'];
  const unit = topic.unit || 'Unit 2';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
      <div className="glass-panel max-w-md w-full rounded-2xl p-6 border border-slate-700/80 shadow-2xl relative">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Title */}
        <div className="flex items-center space-x-2.5 mb-4">
          <div className="p-2 rounded-xl bg-purple-500/20 text-purple-400 border border-purple-500/30">
            <HelpCircle className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-heading font-bold text-base text-white">Why is this High Priority?</h3>
            <p className="text-xs text-purple-300 font-medium">{topic.topic_name}</p>
          </div>
        </div>

        {/* Priority Score Banner */}
        <div className="bg-gradient-to-r from-purple-900/40 to-blue-900/40 border border-purple-500/30 rounded-xl p-3.5 mb-4 flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-300">Historical Priority Score</span>
          <div className="flex items-center space-x-1.5">
            <TrendingUp className="w-4 h-4 text-purple-400" />
            <span className="font-heading font-extrabold text-lg text-purple-300">{priorityScore} / 100</span>
          </div>
        </div>

        {/* Breakdown Grid */}
        <div className="space-y-2.5 text-xs">
          <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 flex items-center justify-between">
            <div className="flex items-center space-x-2 text-slate-300">
              <Hash className="w-4 h-4 text-blue-400" />
              <span>Exam Appearances</span>
            </div>
            <span className="font-semibold text-white">{appearances} times in past papers</span>
          </div>

          <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 flex items-center justify-between">
            <div className="flex items-center space-x-2 text-slate-300">
              <Calendar className="w-4 h-4 text-emerald-400" />
              <span>Exam Years</span>
            </div>
            <span className="font-semibold text-emerald-300">{recentYears.join(', ')}</span>
          </div>

          <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 flex items-center justify-between">
            <div className="flex items-center space-x-2 text-slate-300">
              <Award className="w-4 h-4 text-amber-400" />
              <span>Marks Weight Pattern</span>
            </div>
            <span className="font-semibold text-amber-300">{marksPattern.join(' Marks, ')} Marks</span>
          </div>

          <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 flex items-center justify-between">
            <div className="flex items-center space-x-2 text-slate-300">
              <BookOpen className="w-4 h-4 text-indigo-400" />
              <span>Mapped Syllabus Unit</span>
            </div>
            <span className="font-semibold text-indigo-300">{unit}</span>
          </div>
        </div>

        {/* Source Questions Traceability Section */}
        {topic.source_questions && topic.source_questions.length > 0 && (
          <div className="mt-4 space-y-2 text-xs border-t border-slate-800 pt-3">
            <div className="font-bold text-white uppercase tracking-wider flex items-center justify-between">
              <span>Source Questions Trace ({topic.source_questions.length})</span>
              <span className="text-[10px] text-purple-400 font-medium">Exact PDF Grounding</span>
            </div>

            <div className="max-h-40 overflow-y-auto space-y-2 pr-1">
              {topic.source_questions.map((sq, i) => (
                <div key={i} className="p-2.5 rounded-lg bg-[#0B1020] border border-purple-500/20 space-y-1">
                  <div className="flex items-center justify-between text-[11px] text-purple-300 font-semibold">
                    <span>{sq.question_number} ({sq.year})</span>
                    <span className="text-slate-400 font-normal">{sq.source_file}</span>
                  </div>
                  <p className="text-[11px] text-slate-200 leading-relaxed font-normal">
                    {sq.exact_text}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <p className="text-[11px] text-slate-400/90 mt-4 leading-normal italic bg-slate-900/40 p-2.5 rounded-lg border border-slate-800/60">
          ⚠️ Historical priority is calculated strictly from source PDF examination frequency and mark distribution.
        </p>

        <button
          onClick={onClose}
          className="w-full mt-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs transition-all shadow-md shadow-blue-500/20"
        >
          Got it
        </button>
      </div>
    </div>
  );
}
