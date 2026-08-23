import React from 'react';
import { X, HelpCircle, Calendar, Hash, Award, BookOpen, TrendingUp, ShieldCheck, CheckCircle2 } from 'lucide-react';

export default function WhyPriorityModal({ topic, onClose, onViewEvidence }) {
  if (!topic) return null;

  const title = topic.title || topic.topic_name || topic.question_title || 'Priority Item';
  const priorityScore = topic.priority_score;
  const scoreLabel = priorityScore == null || Number.isNaN(Number(priorityScore)) ? '—' : `${priorityScore} / 100`;
  const signals = topic.signals || {};
  const explanation = topic.explanation || [];
  const sourceQuestions = topic.source_questions || [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
      <div className="saas-card max-w-lg w-full rounded-2xl p-6 relative max-h-[90vh] overflow-y-auto">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-lg text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="flex items-center space-x-2.5 mb-4 pr-8">
          <div className="p-2 rounded-xl bg-purple-500/20 text-purple-600 dark:text-purple-400 border border-purple-500/30 flex-shrink-0">
            <HelpCircle className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-heading font-bold text-base text-slate-900 dark:text-white">Why is this High Priority?</h3>
            <p className="text-xs text-purple-600 dark:text-purple-300 font-medium truncate max-w-xs">{title}</p>
          </div>
        </div>

        {/* Priority Score Banner */}
        <div className="bg-gradient-to-r from-purple-100/80 to-indigo-50/80 dark:from-purple-900/40 dark:to-indigo-900/40 border border-slate-200 dark:border-purple-500/30 rounded-xl p-3.5 mb-4 flex items-center justify-between">
          <div>
            <span className="text-xs font-semibold text-slate-700 dark:text-slate-300 block">Deterministic Evidence Score</span>
            <span className="text-[10px] text-purple-600 dark:text-purple-300 font-medium">Calculated from canonical PDF records</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <TrendingUp className="w-5 h-5 text-purple-600 dark:text-purple-400" />
            <span className="font-heading font-extrabold text-xl text-purple-700 dark:text-purple-300">{scoreLabel}</span>
          </div>
        </div>

        {/* Explanation Bullets */}
        {explanation.length > 0 && (
          <div className="mb-4 p-3 rounded-xl bg-slate-50 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1F2937] space-y-1.5 text-xs">
            <div className="font-bold text-purple-600 dark:text-purple-300 text-[11px] uppercase tracking-wider mb-1">
              Evidence Highlights
            </div>
            {explanation.map((exp, i) => (
              <div key={i} className="text-slate-700 dark:text-slate-300 flex items-start space-x-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-0.5" />
                <span>{exp}</span>
              </div>
            ))}
          </div>
        )}

        {/* Signals Score Breakdown Grid */}
        <div className="space-y-2 text-xs mb-4">
          <div className="font-bold text-slate-900 dark:text-white uppercase tracking-wider text-[11px] mb-1">
            Score Component Signals
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-0.5">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Frequency Weight</span>
              <span className="font-semibold text-slate-900 dark:text-white">{signals.frequency_score ?? '—'} / 22</span>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-0.5">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Year Recurrence Weight</span>
              <span className="font-semibold text-emerald-600 dark:text-emerald-300">{signals.year_recurrence_score ?? '—'} / 26</span>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-0.5">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Exact Repeats Weight</span>
              <span className="font-semibold text-rose-600 dark:text-rose-300">{signals.exact_repeat_score ?? '—'} / 16</span>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-0.5">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Semantic Repeats Weight</span>
              <span className="font-semibold text-amber-600 dark:text-amber-300">{signals.semantic_repeat_score ?? '—'} / 12</span>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-0.5">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Marks Weight</span>
              <span className="font-semibold text-purple-600 dark:text-purple-300">{signals.marks_score ?? '—'} / 10</span>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-0.5">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Recency Weight</span>
              <span className="font-semibold text-blue-600 dark:text-blue-300">{signals.recency_score ?? '—'} / 10</span>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-0.5">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Cross-Year Consistency</span>
              <span className="font-semibold text-teal-600 dark:text-teal-300">{signals.consistency_score ?? '—'} / 8</span>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-0.5">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Syllabus Mapping</span>
              <span className="font-semibold text-indigo-600 dark:text-indigo-300">{signals.syllabus_score ?? '—'} / 6</span>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-0.5">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Source Confidence</span>
              <span className="font-semibold text-slate-800 dark:text-slate-200">{signals.confidence_score ?? '—'} / 6</span>
            </div>
          </div>
        </div>

        {/* Source Traceability Link */}
        {onViewEvidence && sourceQuestions.length > 0 && (
          <button
            onClick={() => {
              onClose();
              onViewEvidence(topic);
            }}
            className="w-full py-2.5 rounded-xl bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 text-purple-700 dark:text-purple-300 font-semibold text-xs transition-all flex items-center justify-center space-x-2 mb-3"
          >
            <ShieldCheck className="w-4 h-4" />
            <span>View All {sourceQuestions.length} Source Question Records →</span>
          </button>
        )}

        {/* Disclaimer */}
        <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-normal italic bg-slate-50 dark:bg-slate-900/40 p-2.5 rounded-lg border border-slate-200 dark:border-slate-800/60">
          ⚠️ Historical priority is calculated strictly from source PDF examination frequency, distinct exam years, and mark distribution.
        </p>

        <button
          onClick={onClose}
          className="w-full mt-4 py-2.5 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-medium text-xs transition-all shadow-md shadow-purple-500/20"
        >
          Got it
        </button>
      </div>
    </div>
  );
}
