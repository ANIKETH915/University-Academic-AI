import React from 'react';
import { HelpCircle, BookOpen, Calendar, Award, CheckCircle2, TrendingUp, ShieldCheck, FileText } from 'lucide-react';

export default function PriorityCard({ rank, item, onWhy, onViewEvidence, onStudy }) {
  if (!item) return null;

  const title = item.title || item.topic_name || item.question_title || 'Priority Item';
  const score = item.priority_score;
  const scoreLabel = score == null || Number.isNaN(Number(score)) ? '—' : `${score} / 100`;
  const tier = item.tier || (rank <= 2 ? 'HIGH' : rank <= 4 ? 'MEDIUM' : 'LOWER');
  const studyLabel = item.study_order_label;
  
  const tierBadge = item.tier_badge
    ? { label: item.tier_badge, bg: tier === 'HIGH' ? 'bg-rose-500/10 text-rose-300 border-rose-500/30' : tier === 'MEDIUM' ? 'bg-amber-500/10 text-amber-300 border-amber-500/30' : 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30' }
    : tier === 'HIGH'
    ? { label: '🔴 HIGH PRIORITY', bg: 'bg-rose-500/10 text-rose-300 border-rose-500/30' }
    : tier === 'MEDIUM'
    ? { label: '🟠 MEDIUM PRIORITY', bg: 'bg-amber-500/10 text-amber-300 border-amber-500/30' }
    : { label: '🟢 LOWER PRIORITY', bg: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30' };

  const unit = item.unit || item.syllabus_unit || 'Unmapped';
  const explanation = item.explanation || [];
  const whySummary = item.why || item.recommendation;
  const sourceQuestions = item.source_questions || [];

  return (
    <div className="saas-card saas-card-hover p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div className="space-y-2 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          {rank && (
            <span className="w-6 h-6 rounded-lg bg-purple-500/20 text-purple-300 flex items-center justify-center font-bold text-xs">
              #{rank}
            </span>
          )}
          <span className={`text-[10px] font-extrabold px-2.5 py-0.5 rounded-md border ${tierBadge.bg}`}>
            {tierBadge.label}
          </span>

          <span className={`text-xs font-semibold px-2 py-0.5 rounded ${unit !== 'Unmapped' ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30' : 'bg-slate-800 text-slate-400 border border-slate-700'}`}>
            {unit}
          </span>

          {studyLabel && (
            <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-slate-800 text-slate-200 border-slate-600">
              {studyLabel}
            </span>
          )}
          <span className="text-xs text-slate-400">
            • Priority Score: <strong className="text-white font-mono">{scoreLabel}</strong>
          </span>
        </div>

        <h4 className="font-heading font-bold text-base text-white">{title}</h4>
        {(item.original_question || item.sample_question) && (
          <p className="text-[11px] text-slate-300 leading-relaxed">
            <span className="text-slate-500 font-semibold uppercase tracking-wide">Original question: </span>
            {item.original_question || item.sample_question}
          </p>
        )}
        
        {whySummary && (
          <p className="text-xs text-slate-300 font-medium leading-relaxed bg-[#0B1020] p-2 rounded-lg border border-[#1F2937]">
            💡 <strong className="text-purple-300">Why ranked here:</strong> {whySummary}
          </p>
        )}

        {/* Explainability bullets */}
        {explanation.length > 0 && (
          <div className="bg-[#080B14] p-2.5 rounded-lg border border-[#1F2937] text-[11px] space-y-1">
            <div className="font-semibold text-purple-300 flex items-center space-x-1">
              <TrendingUp className="w-3 h-3 text-purple-400" />
              <span>Evidence Breakdown</span>
            </div>
            {explanation.slice(0, 3).map((exp, i) => (
              <div key={i} className="text-slate-300 flex items-center space-x-1.5">
                <CheckCircle2 className="w-3 h-3 text-emerald-400 flex-shrink-0" />
                <span>{exp}</span>
              </div>
            ))}
          </div>
        )}

        {/* Grounding count */}
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-400 pt-1">
          <span className="flex items-center space-x-1">
            <ShieldCheck className="w-3.5 h-3.5 text-purple-400" />
            <span>Grounded Evidence: <strong className="text-white">{sourceQuestions.length} question record(s)</strong></span>
          </span>
          {item.typical_marks && (
            <>
              <span>•</span>
              <span className="flex items-center space-x-1">
                <Award className="w-3.5 h-3.5 text-amber-400" />
                <span>Marks: <strong className="text-amber-300">{item.typical_marks}M</strong></span>
              </span>
            </>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 flex-shrink-0">
        {onWhy && (
          <button
            onClick={() => onWhy(item)}
            className="px-3 py-2 rounded-xl bg-[#0B1020] hover:bg-[#1F2937] border border-[#1F2937] text-slate-300 text-xs font-semibold flex items-center space-x-1.5 transition-colors"
            title="View detailed score signals breakdown"
          >
            <HelpCircle className="w-3.5 h-3.5 text-purple-400" />
            <span>Signals</span>
          </button>
        )}

        {onViewEvidence && (
          <button
            onClick={() => onViewEvidence(item)}
            className="px-3 py-2 rounded-xl bg-[#0B1020] hover:bg-[#1F2937] border border-[#1F2937] text-indigo-300 text-xs font-semibold flex items-center space-x-1.5 transition-colors"
            title="View original PDF question text & paper source"
          >
            <FileText className="w-3.5 h-3.5 text-indigo-400" />
            <span>View Evidence ({sourceQuestions.length})</span>
          </button>
        )}

        {onStudy && (
          <button
            onClick={() => onStudy(title)}
            className="px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-md shadow-purple-500/20 flex items-center space-x-1.5 transition-all"
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>Study Topic →</span>
          </button>
        )}
      </div>
    </div>
  );
}
