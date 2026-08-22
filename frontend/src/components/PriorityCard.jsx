import React from 'react';
import { HelpCircle, BookOpen, Calendar, Award, CheckCircle2, TrendingUp } from 'lucide-react';

export default function PriorityCard({ rank, topic, onWhy, onStudy }) {
  const isHigh = rank <= 2;
  const isMedium = rank > 2 && rank <= 4;
  
  const tierBadge = isHigh
    ? { label: '🔴 HIGH PRIORITY', bg: 'bg-rose-500/10 text-rose-300 border-rose-500/30' }
    : isMedium
    ? { label: '🟠 MEDIUM PRIORITY', bg: 'bg-amber-500/10 text-amber-300 border-amber-500/30' }
    : { label: '🟢 LOWER PRIORITY', bg: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30' };

  const predScore = topic.prediction_score || topic.priority_score || 85.0;
  const confidence = topic.prediction_confidence || 'HIGH';
  const explanation = topic.explanation || [];

  return (
    <div className="saas-card saas-card-hover p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div className="space-y-2 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="w-6 h-6 rounded-lg bg-purple-500/20 text-purple-300 flex items-center justify-center font-bold text-xs">
            #{rank}
          </span>
          <span className={`text-[10px] font-extrabold px-2.5 py-0.5 rounded-md border ${tierBadge.bg}`}>
            {tierBadge.label}
          </span>
          <span className="text-[10px] font-bold px-2 py-0.5 rounded-md bg-purple-500/20 text-purple-300 border border-purple-500/30">
            {confidence} CONFIDENCE
          </span>
          <span className="text-xs font-semibold text-purple-300">{topic.unit || 'Unit 2'}</span>
          <span className="text-xs text-slate-500">• Prediction Score: <strong className="text-white">{predScore} / 100</strong></span>
        </div>

        <h4 className="font-heading font-bold text-base text-white">{topic.topic_name}</h4>
        
        {topic.recommendation && (
          <p className="text-xs text-slate-400">{topic.recommendation}</p>
        )}

        {/* Explainability bullets */}
        {explanation.length > 0 && (
          <div className="bg-[#0B1020] p-2.5 rounded-lg border border-[#1F2937] text-[11px] space-y-1">
            <div className="font-semibold text-purple-300 flex items-center space-x-1">
              <TrendingUp className="w-3 h-3 text-purple-400" />
              <span>Why this priority score?</span>
            </div>
            {explanation.slice(0, 3).map((exp, i) => (
              <div key={i} className="text-slate-300 flex items-center space-x-1.5">
                <CheckCircle2 className="w-3 h-3 text-emerald-400 flex-shrink-0" />
                <span>{exp}</span>
              </div>
            ))}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-400 pt-1">
          <span className="flex items-center space-x-1">
            <Award className="w-3.5 h-3.5 text-purple-400" />
            <span>Appearances: <strong className="text-white">{topic.total_appearances || topic.appearances_count || 4} times</strong></span>
          </span>
          <span>•</span>
          <span className="flex items-center space-x-1">
            <Calendar className="w-3.5 h-3.5 text-emerald-400" />
            <span>Years: <strong className="text-emerald-300">{(topic.recent_years || topic.years_appeared || ['2023', '2024']).join(', ')}</strong></span>
          </span>
          <span>•</span>
          <span>Pattern: <strong className="text-amber-300">{(topic.marks_pattern || topic.marks_distribution || ['10']).join('M, ')} Marks</strong></span>
        </div>
      </div>

      <div className="flex items-center space-x-2.5 flex-shrink-0">
        {onWhy && (
          <button
            onClick={() => onWhy(topic)}
            className="px-3 py-2 rounded-xl bg-[#0B1020] hover:bg-[#1F2937] border border-[#1F2937] text-slate-300 text-xs font-semibold flex items-center space-x-1.5 transition-colors"
          >
            <HelpCircle className="w-3.5 h-3.5 text-purple-400" />
            <span>Signals</span>
          </button>
        )}

        {onStudy && (
          <button
            onClick={() => onStudy(topic.topic_name)}
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
