import React from 'react';
import { Award, BookOpen, FileText, ExternalLink, AlertTriangle, HelpCircle, CheckCircle } from 'lucide-react';

export default function ExamAnswerCard({ response }) {
  if (!response) return null;

  // Safe Input Normalization
  const safeQuestion = typeof response.question === 'string' ? response.question : '';
  const safeMode = typeof response.mode === 'string' ? response.mode : 'general';
  const safeAnswer = typeof response.answer === 'string' ? response.answer : '';
  const topScore = typeof response.top_score === 'number' ? response.top_score : (typeof response.top_score === 'string' ? parseFloat(response.top_score) || 0 : 0);
  const isGuardTriggered = Boolean(response.hallucination_guard_triggered) || safeAnswer.startsWith('NOT_FOUND');
  const isClarification = Boolean(response.clarification_requested) || safeAnswer.includes('I need a specific topic') || response.intent === 'ambiguous_topic' || response.intent === 'study_guidance';

  const pyqFrequency = response.pyq_frequency || {};
  const timesAsked = typeof pyqFrequency.times_asked === 'number' ? pyqFrequency.times_asked : 0;
  const pyqYears = Array.isArray(pyqFrequency.years) ? pyqFrequency.years : [];
  const pyqMarks = Array.isArray(pyqFrequency.marks) ? pyqFrequency.marks : [];
  const pyqSummary = typeof pyqFrequency.frequency_summary === 'string' ? pyqFrequency.frequency_summary : '';

  const safeCitations = Array.isArray(response.citations) ? response.citations.filter(c => c && (c.source_file || c.citation_str)) : [];

  // Parse markdown content cleanly without undefined crashes
  const parseSections = (rawText) => {
    const text = typeof rawText === 'string' ? rawText : '';
    if (!text.trim()) return [];
    
    const rawParagraphs = text.split(/\n\n+/);
    return rawParagraphs.map((para) => {
      let title = null;
      let body = typeof para === 'string' ? para.trim() : '';

      if (body.startsWith('#')) {
        const lines = body.split('\n');
        const firstLine = typeof lines[0] === 'string' ? lines[0] : '';
        title = firstLine.replace(/#+/g, '').trim();
        body = lines.slice(1).join('\n').trim();
      } else if (body.includes('\n')) {
        const lines = body.split('\n');
        const firstLine = typeof lines[0] === 'string' ? lines[0] : '';
        if (firstLine.endsWith(':') || (firstLine.length > 0 && firstLine.length < 40 && firstLine.toUpperCase() === firstLine)) {
          title = firstLine.replace(':', '').trim();
          body = lines.slice(1).join('\n').trim();
        }
      }

      body = body.replace(/\*\*/g, '').replace(/__/g, '');
      return { title, body };
    });
  };

  const sections = parseSections(safeAnswer);

  const renderFormattedContent = (text) => {
    const safeContent = typeof text === 'string' ? text : '';
    const lines = safeContent.split('\n');
    const elements = [];

    lines.forEach((line, idx) => {
      const trimmed = typeof line === 'string' ? line.trim() : '';
      if (!trimmed) return;

      if (trimmed.startsWith('|')) {
        elements.push(
          <div key={idx} className="font-mono text-xs p-2 my-1 rounded bg-slate-100 dark:bg-[#080B14] border border-slate-200 dark:border-[#1F2937] text-purple-700 dark:text-purple-200 overflow-x-auto">
            {trimmed}
          </div>
        );
      } else if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
        elements.push(
          <div key={idx} className="flex items-start space-x-2 my-1.5 pl-2">
            <span className="text-purple-600 dark:text-purple-400 font-bold">•</span>
            <span className="text-slate-800 dark:text-slate-200 text-xs sm:text-sm">{trimmed.substring(2)}</span>
          </div>
        );
      } else if (/^\d+\./.test(trimmed)) {
        const match = trimmed.match(/^(\d+\.)\s*(.*)/);
        elements.push(
          <div key={idx} className="flex items-start space-x-2 my-1.5 pl-2">
            <span className="text-purple-600 dark:text-purple-400 font-bold text-xs">{match ? match[1] : '1.'}</span>
            <span className="text-slate-800 dark:text-slate-200 text-xs sm:text-sm">{match ? match[2] : trimmed}</span>
          </div>
        );
      } else {
        elements.push(
          <p key={idx} className="my-1.5 text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
            {trimmed}
          </p>
        );
      }
    });

    return elements;
  };

  const modeBadgeText = safeMode.replace(/_/g, ' ').toUpperCase();

  // CASE B: Clarification / Ambiguous Query Response
  if (isClarification) {
    return (
      <div className="saas-card p-6 space-y-4 border-amber-500/30 bg-amber-500/10 animate-in fade-in">
        <div className="flex items-center space-x-3 text-amber-600 dark:text-amber-400 pb-3 border-b border-slate-200 dark:border-[#1F2937]">
          <div className="p-2 rounded-xl bg-amber-500/20 text-amber-700 dark:text-amber-300 border border-amber-500/30">
            <HelpCircle className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-heading font-extrabold text-base text-slate-900 dark:text-white">Specific Topic Required</h3>
            <p className="text-xs text-slate-600 dark:text-slate-400">Please refine your question or specify an academic topic.</p>
          </div>
        </div>

        <div className="text-xs sm:text-sm text-slate-800 dark:text-slate-300 space-y-2 leading-relaxed">
          {renderFormattedContent(safeAnswer)}
        </div>
      </div>
    );
  }

  // CASE C: NOT_FOUND / Hallucination Guard Triggered
  if (isGuardTriggered) {
    return (
      <div className="saas-card p-6 space-y-4 border-rose-500/30 bg-rose-500/10 animate-in fade-in">
        <div className="flex items-center space-x-3 text-rose-600 dark:text-rose-400 pb-3 border-b border-slate-200 dark:border-[#1F2937]">
          <div className="p-2 rounded-xl bg-rose-500/20 text-rose-700 dark:text-rose-300 border border-rose-500/30">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-heading font-extrabold text-base text-slate-900 dark:text-white">Topic Not Found in Uploaded Documents</h3>
            <p className="text-xs text-slate-600 dark:text-slate-400">The uploaded documents do not contain reliable information for this query.</p>
          </div>
        </div>

        <p className="text-xs sm:text-sm text-slate-800 dark:text-slate-300 leading-relaxed">
          {safeAnswer.replace('NOT_FOUND:', '').trim() || 'That information was not found in the documents uploaded to this academic workspace.'}
        </p>
      </div>
    );
  }

  // CASE A: Normal Grounded Exam Response
  return (
    <div className="saas-card p-6 space-y-6 animate-in fade-in">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200 dark:border-[#1F2937]">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-[10px] font-extrabold uppercase tracking-wider px-2.5 py-0.5 rounded bg-purple-500/20 text-purple-700 dark:text-purple-300 border border-purple-500/30">
              {modeBadgeText} EXAM RESPONSE
            </span>
            <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
              Grounding Score: {topScore.toFixed(3)}
            </span>
          </div>
          {safeQuestion && (
            <h3 className="font-heading font-extrabold text-xl text-slate-900 dark:text-white mt-1.5">{safeQuestion}</h3>
          )}
        </div>
      </div>

      {/* Sections */}
      <div className="space-y-4">
        {sections.map((sec, idx) => (
          <div key={idx} className="p-4 rounded-xl bg-slate-50 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1F2937] space-y-2">
            {sec.title && (
              <h4 className="font-heading font-bold text-xs text-purple-600 dark:text-purple-300 uppercase tracking-wider border-b border-slate-200 dark:border-[#1F2937] pb-1.5">
                {sec.title}
              </h4>
            )}
            <div className="pt-0.5">
              {renderFormattedContent(sec.body)}
            </div>
          </div>
        ))}
      </div>

      {/* PYQ Evidence Card */}
      {timesAsked > 0 && (
        <div className="p-4 rounded-xl bg-purple-950/20 border border-purple-500/30 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-purple-500/20 text-purple-300 border border-purple-500/30">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <div className="font-heading font-bold text-white text-sm">Past Exam PYQ Evidence</div>
              <div className="text-purple-300 text-xs">{pyqSummary}</div>
            </div>
          </div>
          <div className="flex items-center space-x-2 text-[11px]">
            {pyqYears.length > 0 && (
              <span className="px-2.5 py-1 rounded-lg bg-purple-900/40 text-purple-200 font-semibold border border-purple-500/30">
                Years: {pyqYears.join(', ')}
              </span>
            )}
            {pyqMarks.length > 0 && (
              <span className="px-2.5 py-1 rounded-lg bg-amber-900/40 text-amber-200 font-semibold border border-amber-500/30">
                Marks: {pyqMarks.join(', ')} Marks
              </span>
            )}
          </div>
        </div>
      )}

      {/* Isolated Source Citations */}
      {safeCitations.length > 0 && (
        <div className="pt-4 border-t border-[#1F2937]">
          <h4 className="font-heading font-bold text-xs text-slate-400 uppercase tracking-wider mb-3 flex items-center space-x-2">
            <BookOpen className="w-4 h-4 text-purple-400" />
            <span>Isolated Source Citations</span>
          </h4>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            {safeCitations.map((cit, cIdx) => {
              const isSyllabus = cit.type === 'syllabus';
              const sourceFile = typeof cit.source_file === 'string' ? cit.source_file : 'Academic Document';
              const sourcePage = typeof cit.source_page === 'number' || typeof cit.source_page === 'string' ? cit.source_page : 1;

              return (
                <div
                  key={cIdx}
                  className={`p-3.5 rounded-xl border flex items-center justify-between ${
                    isSyllabus
                      ? 'bg-blue-950/30 border-blue-500/30 text-blue-200'
                      : 'bg-purple-950/30 border-purple-500/30 text-purple-200'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${isSyllabus ? 'bg-blue-500/20 text-blue-400' : 'bg-purple-500/20 text-purple-400'}`}>
                      {isSyllabus ? <FileText className="w-4 h-4" /> : <Award className="w-4 h-4" />}
                    </div>
                    <div>
                      <div className="font-bold text-white text-xs">
                        {isSyllabus ? '📚 Syllabus Source' : '📝 PYQ Source'}
                      </div>
                      <div className="text-[11px] opacity-80">{sourceFile} (Page {sourcePage})</div>
                    </div>
                  </div>
                  <ExternalLink className="w-3.5 h-3.5 opacity-60 flex-shrink-0" />
                </div>
              );
            })}
          </div>
        </div>
      )}

    </div>
  );
}
