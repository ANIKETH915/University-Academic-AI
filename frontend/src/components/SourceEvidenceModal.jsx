import React from 'react';
import { X, FileText, Calendar, Award, Hash, CheckCircle2, ShieldCheck, ExternalLink } from 'lucide-react';

export default function SourceEvidenceModal({ item, onClose, onStudy }) {
  if (!item) return null;

  const title = item.title || item.topic_name || item.question_title || 'Source Evidence';
  const questions = item.source_questions || [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
      <div className="saas-card max-w-2xl w-full rounded-2xl p-6 relative max-h-[90vh] flex flex-col">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-lg text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Modal Header */}
        <div className="flex items-center space-x-3 mb-4 pr-8">
          <div className="p-2.5 rounded-xl bg-purple-500/20 text-purple-600 dark:text-purple-400 border border-purple-500/30">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-[10px] font-extrabold px-2 py-0.5 rounded-md bg-purple-500/20 text-purple-700 dark:text-purple-300 border border-purple-500/30 uppercase">
                {item.tier_badge || 'Source Grounded'}
              </span>
              <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">
                Priority Score: {item.priority_score == null ? '—' : `${item.priority_score} / 100`}
              </span>
            </div>
            <h3 className="font-heading font-bold text-lg text-slate-900 dark:text-white mt-0.5">{title}</h3>
          </div>
        </div>

        {/* Info Banner */}
        <div className="p-3 rounded-xl bg-slate-50 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1F2937] text-xs text-slate-700 dark:text-slate-300 mb-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <FileText className="w-4 h-4 text-purple-600 dark:text-purple-400 flex-shrink-0" />
            <span>
              Grounding Evidence: <strong className="text-slate-900 dark:text-white">{questions.length} canonical PDF question record(s)</strong>
            </span>
          </div>
          {item.unit && item.unit !== 'Unmapped' && (
            <span className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-700 dark:text-indigo-300 text-[10px] font-semibold border border-indigo-500/30">
              Syllabus: {item.unit}
            </span>
          )}
        </div>

        {/* Source Questions Traceability List */}
        <div className="flex-1 overflow-y-auto space-y-3 pr-1">
          {questions.length === 0 ? (
            <div className="p-8 text-center text-slate-500 text-xs saas-card">
              No direct question records associated with this item.
            </div>
          ) : (
            questions.map((q, idx) => {
              const relLabel =
                q.relationship === 'EXACT_REPEAT'
                  ? { label: 'Exact Repeat', bg: 'bg-rose-500/10 text-rose-700 dark:text-rose-300 border-rose-500/30' }
                  : q.relationship === 'SEMANTIC_REPEAT'
                  ? { label: 'Semantic Repeat', bg: 'bg-amber-500/10 text-amber-700 dark:text-amber-300 border-amber-500/30' }
                  : { label: 'Topic Member', bg: 'bg-purple-500/10 text-purple-700 dark:text-purple-300 border-purple-500/30' };

              return (
                <div key={idx} className="p-4 rounded-xl bg-slate-50 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1F2937] space-y-2.5 hover:border-purple-500/40 transition-colors">
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                    <div className="flex items-center space-x-2">
                      <span className="font-mono font-bold text-purple-700 dark:text-purple-300 bg-purple-500/15 px-2 py-0.5 rounded">
                        {q.question_id || `Q${idx + 1}`}
                      </span>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${relLabel.bg}`}>
                        {relLabel.label}
                      </span>
                    </div>

                    <div className="flex items-center space-x-3 text-[11px] text-slate-500 dark:text-slate-400">
                      <span className="flex items-center space-x-1">
                        <Calendar className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                        <strong className="text-slate-900 dark:text-white">{q.year || '?'} ({q.exam_session || 'Exam'})</strong>
                      </span>
                      <span>•</span>
                      <span className="flex items-center space-x-1">
                        <Award className="w-3 h-3 text-amber-500 dark:text-amber-400" />
                        <strong className="text-amber-600 dark:text-amber-300">{q.marks || 5} Marks</strong>
                      </span>
                      <span>•</span>
                      <span className="text-slate-500 font-mono">Page {q.source_page || 1}</span>
                    </div>
                  </div>

                  {/* Question Text */}
                  <p className="text-xs text-slate-800 dark:text-slate-200 leading-relaxed bg-white dark:bg-[#080B14] p-3 rounded-lg border border-slate-200 dark:border-slate-800 font-sans">
                    "{q.exact_text}"
                  </p>

                  <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400 pt-0.5">
                    <span className="truncate max-w-xs text-slate-500">
                      PDF: <strong className="text-slate-700 dark:text-slate-300">{q.source_file}</strong>
                    </span>
                    {onStudy && (
                      <button
                        onClick={() => onStudy(q.exact_text)}
                        className="text-purple-600 dark:text-purple-400 hover:text-purple-500 dark:hover:text-purple-300 font-semibold flex items-center space-x-1 transition-colors"
                      >
                        <span>Ask AI about this Q</span>
                        <ExternalLink className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Modal Footer */}
        <div className="mt-4 pt-3 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between text-xs">
          <span className="text-slate-500 dark:text-slate-400 italic text-[11px]">
            Grounded directly in canonical validated PDF question records.
          </span>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold transition-all shadow-md shadow-purple-500/20"
          >
            Close Traceability
          </button>
        </div>
      </div>
    </div>
  );
}
