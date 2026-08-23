import React, { useState } from 'react';
import { useWorkspace } from '../context/WorkspaceContext';
import { askQuestion, fetchPYQAnalysis, fetchStudyPriority } from '../api/academicApi';
import { classifyQueryIntent } from '../utils/intentRouter';
import PageHeader from '../components/PageHeader';
import ExamAnswerCard from '../components/ExamAnswerCard';
import HallucinationWarning from '../components/HallucinationWarning';
import { 
  Send, 
  Cpu, 
  Loader2, 
  Sparkles, 
  GraduationCap,
  Layers,
  Database,
  Upload,
  Flame,
  Award,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

export default function AskPage({ setActiveTab, initialQuestion = '', initialMode = '10_marks' }) {
  const { activeWorkspace, setIsSelectorOpen } = useWorkspace();

  const totalFiles = activeWorkspace.syllabusFiles.length + activeWorkspace.pyqFiles.length;
  const hasDocuments = totalFiles > 0;
  const hasSyllabus = activeWorkspace.syllabusFiles.length > 0;
  const hasPyqs = activeWorkspace.pyqFiles.length > 0;

  const [question, setQuestion] = useState(initialQuestion);
  const [mode, setMode] = useState(initialMode || '10_marks');
  const [loading, setLoading] = useState(false);
  const [activeIntent, setActiveIntent] = useState('GENERAL_QA');
  const [response, setResponse] = useState(null);
  const [pyqAnalysisData, setPyqAnalysisData] = useState(null);
  const [studyPriorityData, setStudyPriorityData] = useState(null);
  const [error, setError] = useState(null);

  // Generic & Dynamic Example Prompts based on document availability
  const quickPrompts = [];
  if (hasSyllabus && hasPyqs) {
    quickPrompts.push({ text: 'Explain a key topic from my syllabus', mode: '5_marks' });
    quickPrompts.push({ text: 'Which questions are repeatedly asked in past papers?', mode: 'general' });
    quickPrompts.push({ text: 'Which topics should I study first for the exam?', mode: 'general' });
    quickPrompts.push({ text: 'Explain a 10-mark exam topic in detail', mode: '10_marks' });
  } else if (hasSyllabus) {
    quickPrompts.push({ text: 'Explain a key topic from my syllabus', mode: '5_marks' });
    quickPrompts.push({ text: 'What core topics are covered in my syllabus?', mode: 'general' });
    quickPrompts.push({ text: 'Explain a core definition for 2 marks', mode: '2_marks' });
  } else if (hasPyqs) {
    quickPrompts.push({ text: 'Which questions are repeatedly asked in past papers?', mode: 'general' });
    quickPrompts.push({ text: 'Which topics should I study first?', mode: 'general' });
    quickPrompts.push({ text: 'Explain a 10-mark exam question from past papers', mode: '10_marks' });
  } else {
    quickPrompts.push({ text: 'Explain a key topic for 5 marks', mode: '5_marks' });
    quickPrompts.push({ text: 'Explain a definition for 2 marks', mode: '2_marks' });
    quickPrompts.push({ text: 'Explain a detailed topic for 10 marks', mode: '10_marks' });
  }

  const handleAsk = async (e) => {
    if (e) e.preventDefault();
    if (!question.trim()) return;

    const detectedIntent = classifyQueryIntent(question);
    setActiveIntent(detectedIntent);
    setLoading(true);
    setError(null);
    setResponse(null);
    setPyqAnalysisData(null);
    setStudyPriorityData(null);

    try {
      if (detectedIntent === 'PYQ_ANALYSIS') {
        const data = await fetchPYQAnalysis(
          activeWorkspace.subject || 'Subject',
          activeWorkspace.semester || 'Semester',
          true,
          activeWorkspace.id
        );
        setPyqAnalysisData(data?.total_valid_questions || data?.total_questions_analyzed ? data : { empty: true, ...data });
      } else if (detectedIntent === 'STUDY_PRIORITY') {
        const data = await fetchStudyPriority(
          activeWorkspace.subject || 'Subject',
          activeWorkspace.semester || 'Semester',
          5,
          true,
          activeWorkspace.id
        );
        setStudyPriorityData((data?.top_high_priority_topics || []).length ? data : { empty: true, ...data });
      } else { // GENERAL_QA
        if (!activeWorkspace?.id) {
          setResponse({
            question: question,
            mode: mode,
            doc_type: 'both',
            answer: 'NOT_FOUND: Select or create an academic workspace first.',
            hallucination_guard_triggered: true,
            top_score: 0.0,
            pyq_frequency: { times_asked: 0, years: [], frequency_summary: 'No workspace selected.' },
            citations: [],
            retrieved_chunks_count: 0
          });
        } else {
          console.log('[RETRIEVAL] workspace_id=', activeWorkspace.id, 'query=', question);
          const data = await askQuestion({
            question: question,
            workspace_id: activeWorkspace.id,
            mode: mode,
            doc_type: 'both',
            subject: activeWorkspace.subject,
            semester: activeWorkspace.semester,
            university: activeWorkspace.university,
            branch: activeWorkspace.branch
          });
          console.log(
            '[ANSWER] workspace_id=', activeWorkspace.id,
            'retrieved=', data?.retrieved_chunks_count,
            'sources=', (data?.citations || []).map(c => c.source_file || c.filename).filter(Boolean)
          );
          setResponse(data);
        }
      }
    } catch (err) {
      setError('Failed to reach backend API. Check if Uvicorn server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const handleQuickPrompt = (promptText, promptMode) => {
    setQuestion(promptText);
    setMode(promptMode);
  };

  return (
    <div className="space-y-6">
      
      {/* Standardized Page Header */}
      <PageHeader
        breadcrumb="Ask AI"
        title={`Ask Academic AI (${activeWorkspace.subject || 'Your Workspace'})`}
        subtitle={`Grounded answer generation strictly scoped to ${activeWorkspace.subject || 'your uploaded documents'}.`}
        icon={Cpu}
        badge="Grounded RAG"
      />

      {/* Target Scope Banner */}
      <div className="p-4 rounded-xl bg-white dark:bg-[#0B1020] border border-slate-200 dark:border-[#1F2937] text-slate-700 dark:text-slate-300 text-xs flex items-center justify-between gap-3 shadow-xs">
        <div className="flex items-center space-x-2.5">
          <div className="p-1.5 rounded-lg bg-purple-500/20 text-purple-600 dark:text-purple-300 font-bold">
            <GraduationCap className="w-4 h-4" />
          </div>
          <div>
            <span className="font-semibold text-slate-900 dark:text-white">Target Scope:</span> Answers generated ONLY from documents uploaded to this workspace.
          </div>
        </div>

        <button
          onClick={() => setIsSelectorOpen(true)}
          className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-[#111827] hover:bg-slate-200 dark:hover:bg-[#1F2937] border border-slate-200 dark:border-[#1F2937] text-purple-600 dark:text-purple-300 text-xs font-semibold flex items-center space-x-1 transition-colors flex-shrink-0"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Switch Workspace</span>
        </button>
      </div>

      {/* Main Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        {/* LEFT: Context Panel */}
        <div className="space-y-4">
          <div className="saas-card p-5 space-y-3">
            <div className="flex items-center space-x-2 text-purple-600 dark:text-purple-400">
              <GraduationCap className="w-4 h-4" />
              <h3 className="font-heading font-bold text-xs uppercase tracking-wider text-slate-700 dark:text-slate-300">Active Context</h3>
            </div>

            <div className="space-y-2 text-xs">
              <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1F2937]">
                <div className="text-[10px] text-slate-500 font-semibold uppercase">University</div>
                <div className="font-semibold text-slate-900 dark:text-white mt-0.5">{activeWorkspace.university || 'Not specified'}</div>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1F2937]">
                <div className="text-[10px] text-slate-500 font-semibold uppercase">Program & Semester</div>
                <div className="font-semibold text-purple-600 dark:text-purple-300 mt-0.5">{activeWorkspace.branch || 'Not specified'} · {activeWorkspace.semester || 'Not specified'}</div>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1F2937]">
                <div className="text-[10px] text-slate-500 font-semibold uppercase">Subject Workspace</div>
                <div className="font-semibold text-blue-600 dark:text-blue-300 mt-0.5">{activeWorkspace.subject || 'Not specified'} {activeWorkspace.subjectCode ? `(${activeWorkspace.subjectCode})` : ''}</div>
              </div>
            </div>
          </div>

          {/* Quick Prompts Panel */}
          <div className="saas-card p-4 space-y-2">
            <div className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider flex items-center space-x-1.5">
              <Sparkles className="w-3.5 h-3.5 text-amber-500 dark:text-amber-400" />
              <span>Example Questions</span>
            </div>
            <div className="space-y-1.5 pt-1">
              {quickPrompts.map((p, idx) => (
                <button
                  key={idx}
                  onClick={() => handleQuickPrompt(p.text, p.mode)}
                  className="w-full text-left p-2 rounded-lg bg-slate-50 dark:bg-[#0B1020] hover:bg-slate-100 dark:hover:bg-[#1F2937] text-slate-700 dark:text-slate-300 text-[11px] border border-slate-200 dark:border-[#1F2937] transition-colors truncate"
                >
                  {p.text}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* RIGHT: Answer Workspace Canvas */}
        <div className="lg:col-span-3 space-y-4">
          
          {/* Question Input Card */}
          <div className="saas-card p-5 space-y-4">
            
            {/* Format Mode Selectors */}
            <div>
              <label className="block text-[11px] font-semibold text-slate-500 dark:text-slate-400 mb-2 uppercase tracking-wider">Answer Format Mode</label>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {[
                  { id: 'general', label: 'General Mode' },
                  { id: '2_marks', label: '2 Marks' },
                  { id: '5_marks', label: '5 Marks' },
                  { id: '10_marks', label: '10 Marks (Detailed)', highlight: true }
                ].map(m => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setMode(m.id)}
                    className={`px-3 py-2 rounded-xl text-xs font-semibold border transition-all text-center ${
                      mode === m.id
                        ? m.highlight 
                          ? 'bg-gradient-to-r from-purple-600 to-indigo-600 border-purple-400 text-white shadow-lg glow-purple'
                          : 'bg-purple-600 border-purple-500 text-white'
                        : 'bg-slate-50 dark:bg-[#0B1020] border-slate-200 dark:border-[#1F2937] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Input Form */}
            <form onSubmit={handleAsk} className="relative">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder={`Ask anything about ${activeWorkspace.subject || 'your documents'}...`}
                className="w-full bg-slate-50 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1F2937] rounded-xl pl-4 pr-14 py-3.5 text-sm text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30"
              />
              <button
                type="submit"
                disabled={loading}
                className="absolute right-2 top-2 px-3 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs transition-all shadow-md shadow-purple-500/20 flex items-center space-x-1"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
              </button>
            </form>
          </div>

          {/* Error Banner */}
          {error && (
            <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-600 dark:text-rose-300 text-xs">
              ⚠️ {error}
            </div>
          )}

          {/* INTENT-BASED UI RENDERING */}
          {activeIntent === 'PYQ_ANALYSIS' && pyqAnalysisData && (
            <div className="saas-card p-6 space-y-4">
              <div className="flex items-center space-x-2 border-b border-slate-200 dark:border-[#1F2937] pb-3 text-purple-600 dark:text-purple-400">
                <Flame className="w-5 h-5" />
                <div>
                  <h3 className="font-heading font-extrabold text-base text-slate-900 dark:text-white">PYQ INTELLIGENCE — RECURRING EXAM QUESTIONS</h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400">Question pattern frequency across uploaded previous-year papers.</p>
                </div>
              </div>

              {pyqAnalysisData.unavailable ? (
                <div className="p-6 rounded-xl bg-rose-500/10 border border-rose-500/30 text-center space-y-2">
                  <div className="text-xs text-rose-600 dark:text-rose-200 font-semibold">PYQ analysis could not be loaded.</div>
                  <div className="text-xs text-slate-600 dark:text-slate-300">{pyqAnalysisData.error || 'The backend rejected the request.'}</div>
                  <div className="text-[11px] text-slate-500">This is a backend error, not an empty workspace.</div>
                </div>
              ) : pyqAnalysisData.empty || !pyqAnalysisData.topics || pyqAnalysisData.topics.length === 0 ? (
                <div className="p-6 rounded-xl bg-purple-500/10 border border-purple-500/30 text-center space-y-3">
                  <div className="text-xs text-slate-700 dark:text-slate-300 font-semibold">No previous-year papers have been uploaded to this workspace yet.</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">Upload PYQ PDFs to discover recurring questions and exam frequency patterns.</div>
                  <button
                    onClick={() => setActiveTab && setActiveTab('workspace')}
                    className="px-4 py-2 rounded-xl bg-purple-600 text-white font-semibold text-xs inline-flex items-center space-x-1.5"
                  >
                    <Upload className="w-3.5 h-3.5" />
                    <span>Upload PYQs</span>
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  {pyqAnalysisData.topics.map((t, idx) => (
                    <div key={idx} className="p-4 rounded-xl bg-slate-50 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1F2937] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="w-6 h-6 rounded-lg bg-purple-600/20 text-purple-700 dark:text-purple-300 font-extrabold text-xs flex items-center justify-center border border-purple-500/30">
                            0{idx+1}
                          </span>
                          <span className="font-heading font-bold text-sm text-slate-900 dark:text-white">{t.topic_name}</span>
                        </div>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{t.sample_question || t.topic_name}</p>
                      </div>
                      <div className="flex items-center space-x-2 text-xs flex-shrink-0">
                        <span className="px-2.5 py-1 rounded-lg bg-purple-500/10 text-purple-700 dark:text-purple-200 font-semibold border border-purple-500/30">
                          Asked {t.unique_occurrence_count ?? t.appearances_count} times
                        </span>
                        <span className="px-2.5 py-1 rounded-lg bg-emerald-500/10 text-emerald-700 dark:text-emerald-200 font-semibold border border-emerald-500/30">
                          {t.years_appeared ? t.years_appeared.join(', ') : 'Recent'}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeIntent === 'STUDY_PRIORITY' && studyPriorityData && (
            <div className="saas-card p-6 space-y-4">
              <div className="flex items-center space-x-2 border-b border-slate-200 dark:border-[#1F2937] pb-3 text-emerald-600 dark:text-emerald-400">
                <GraduationCap className="w-5 h-5" />
                <div>
                  <h3 className="font-heading font-extrabold text-base text-slate-900 dark:text-white">STUDY PRIORITY RECOMMENDATIONS</h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400">Ranked exam topics based on historical frequency and recency weighting.</p>
                </div>
              </div>

              {studyPriorityData.unavailable ? (
                <div className="p-6 rounded-xl bg-rose-500/10 border border-rose-500/30 text-center space-y-2">
                  <div className="text-xs text-rose-600 dark:text-rose-200 font-semibold">Study priority could not be loaded.</div>
                  <div className="text-xs text-slate-600 dark:text-slate-300">{studyPriorityData.error || 'The backend rejected the request.'}</div>
                  <div className="text-[11px] text-slate-500">This is a backend error, not an empty workspace.</div>
                </div>
              ) : studyPriorityData.empty || !studyPriorityData.top_high_priority_topics || studyPriorityData.top_high_priority_topics.length === 0 ? (
                <div className="p-6 rounded-xl bg-purple-500/10 border border-purple-500/30 text-center space-y-3">
                  <div className="text-xs text-slate-700 dark:text-slate-300 font-semibold">No previous-year papers have been uploaded to this workspace yet.</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">Upload PYQ PDFs to generate a focused, priority-ranked study plan.</div>
                  <button
                    onClick={() => setActiveTab && setActiveTab('workspace')}
                    className="px-4 py-2 rounded-xl bg-purple-600 text-white font-semibold text-xs inline-flex items-center space-x-1.5"
                  >
                    <Upload className="w-3.5 h-3.5" />
                    <span>Upload PYQs</span>
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  {studyPriorityData.top_high_priority_topics.map((item, idx) => (
                    <div key={idx} className="p-4 rounded-xl bg-slate-50 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1F2937] space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <span className="px-2.5 py-0.5 rounded-md bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 text-xs font-extrabold border border-emerald-500/30">
                            Rank #{item.rank || idx+1}
                          </span>
                          <span className="font-heading font-bold text-sm text-slate-900 dark:text-white">{item.topic_name}</span>
                        </div>
                        <span className="text-xs font-bold text-purple-600 dark:text-purple-300">
                          Priority Score: {item.priority_score}/100
                        </span>
                      </div>
                      <p className="text-xs text-slate-600 dark:text-slate-400">{item.recommendation}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeIntent === 'GENERAL_QA' && response && (
            <div className="space-y-4">
              {response.hallucination_guard_triggered ? (
                <HallucinationWarning topicName={response.question} />
              ) : (
                <ExamAnswerCard response={response} />
              )}
            </div>
          )}

        </div>

      </div>

    </div>
  );
}
