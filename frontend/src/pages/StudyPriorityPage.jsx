import React, { useState, useEffect } from 'react';
import { useWorkspace } from '../context/WorkspaceContext';
import { fetchPYQAnalysis, fetchStudyPriority } from '../api/academicApi';
import PageHeader from '../components/PageHeader';
import PriorityCard from '../components/PriorityCard';
import WhyPriorityModal from '../components/WhyPriorityModal';
import SourceEvidenceModal from '../components/SourceEvidenceModal';
import { 
  GraduationCap, 
  ShieldAlert, 
  Loader2, 
  Layers, 
  Upload, 
  AlertCircle, 
  TrendingUp, 
  Repeat, 
  Calendar, 
  BookOpen, 
  ShieldCheck, 
  Award,
  Sparkles,
  FileText,
  CheckCircle2,
  RefreshCw,
  PlusCircle,
  AlertTriangle
} from 'lucide-react';

export default function StudyPriorityPage({ setActiveTab, setSelectedTopicForAsk }) {
  const { activeWorkspace, setIsSelectorOpen } = useWorkspace();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeSubTab, setActiveSubTab] = useState('study-first'); // study-first | repeated-questions | recurring-topics | related-topics | modules
  
  // Modals
  const [selectedWhyTopic, setSelectedWhyTopic] = useState(null);
  const [selectedEvidenceItem, setSelectedEvidenceItem] = useState(null);

  const isInvalidWorkspace = !activeWorkspace || !activeWorkspace.id || activeWorkspace.id === '/' || activeWorkspace.id === 'undefined' || activeWorkspace.id === 'null';

  const syllabusFiles = activeWorkspace?.syllabusFiles || [];
  const pyqFiles = activeWorkspace?.pyqFiles || [];
  const hasSyllabus = syllabusFiles.length > 0;
  const hasLocalPyqs = pyqFiles.length > 0;

  async function loadPriority(isRefresh = false) {
    if (isInvalidWorkspace) {
      setData(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    console.log('[STUDY_PRIORITY] workspace_id=', activeWorkspace.id, 'isRefresh=', isRefresh);

    try {
      const res = await fetchPYQAnalysis(
        activeWorkspace.subject || 'Subject',
        activeWorkspace.semester || 'Semester',
        true,
        activeWorkspace.id
      );
      setData(res);
    } catch (err) {
      console.warn('[STUDY_PRIORITY] load error:', err);
    } finally {
      setLoading(false);
    }
  }

  // Clear state & fetch new workspace documents when workspace changes
  useEffect(() => {
    setData(null);
    loadPriority(false);
  }, [activeWorkspace?.id, activeWorkspace?.subject, activeWorkspace?.semester, pyqFiles.length]);

  const handleStudyTopic = (topicName) => {
    if (setSelectedTopicForAsk) {
      setSelectedTopicForAsk(topicName);
    }
    setActiveTab('ask');
  };

  const totalPapers = data?.total_papers || pyqFiles.length || 0;
  const totalQuestions = data?.total_valid_questions || data?.total_questions_analyzed || 0;
  const yearsCovered = data?.years_covered || [];
  const yearsRange = yearsCovered.length > 0
    ? (yearsCovered.length === 1 ? `${yearsCovered[0]}` : `${Math.min(...yearsCovered)}–${Math.max(...yearsCovered)}`)
    : 'N/A';

  const papersList = data?.papers || [];
  const summary = data?.summary_stats || {};
  const recommendedPlan = data?.recommended_study_plan || [];
  const questionPriorities = data?.question_priorities || [];
  const topicPriorities = data?.topic_priorities || [];

  const hasIngestedPapers = totalPapers > 0 || totalQuestions > 0 || hasLocalPyqs;

  return (
    <div className="space-y-6">
      
      {/* Modals */}
      {selectedWhyTopic && (
        <WhyPriorityModal
          topic={selectedWhyTopic}
          onClose={() => setSelectedWhyTopic(null)}
          onViewEvidence={(item) => setSelectedEvidenceItem(item)}
        />
      )}

      {selectedEvidenceItem && (
        <SourceEvidenceModal
          item={selectedEvidenceItem}
          onClose={() => setSelectedEvidenceItem(null)}
          onStudy={(txt) => handleStudyTopic(txt)}
        />
      )}

      {/* Page Header */}
      <PageHeader
        breadcrumb="Exam Intelligence"
        title={`Study Priority Planner (${activeWorkspace?.subject || 'Your Subject'})`}
        subtitle="Automatic study planner consuming ingested PYQ records and recurrence evidence from your workspace."
        icon={GraduationCap}
        badge="Workspace Ingested"
      />
      {/* INVALID WORKSPACE STATE */}
      {isInvalidWorkspace ? (
        <div className="saas-card p-10 text-center space-y-4 max-w-2xl mx-auto my-6">
          <div className="w-16 h-16 rounded-3xl bg-amber-500/10 border border-amber-500/20 text-amber-500 dark:text-amber-400 flex items-center justify-center mx-auto">
            <AlertCircle className="w-8 h-8" />
          </div>

          <div className="space-y-1">
            <h3 className="font-heading font-extrabold text-lg text-slate-900 dark:text-white">
              No active workspace selected
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 max-w-md mx-auto leading-relaxed">
              Please select or create an academic workspace to generate study priorities.
            </p>
          </div>

          <button
            onClick={() => setIsSelectorOpen(true)}
            className="px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-lg shadow-purple-500/20 inline-flex items-center space-x-2 transition-all"
          >
            <Layers className="w-4 h-4" />
            <span>Select Workspace</span>
          </button>
        </div>
      ) : loading ? (
        <div className="p-12 text-center text-slate-500 dark:text-slate-400 saas-card">
          <Loader2 className="w-8 h-8 animate-spin mx-auto text-purple-600 dark:text-purple-400 mb-2" />
          <p className="text-xs">Reading ingested PYQ documents for {activeWorkspace.subject || 'workspace'}...</p>
        </div>
      ) : !hasIngestedPapers ? (
        /* NO PYQ DOCUMENTS IN WORKSPACE AT ALL - SHOW UPLOAD CTA ONLY HERE */
        <div className="saas-card p-10 text-center space-y-4 max-w-2xl mx-auto my-6">
          <div className="w-16 h-16 rounded-3xl bg-purple-500/10 border border-purple-500/20 text-purple-600 dark:text-purple-400 flex items-center justify-center mx-auto">
            <Upload className="w-8 h-8" />
          </div>
          <div className="space-y-1.5">
            <h3 className="font-heading font-extrabold text-lg text-slate-900 dark:text-white">
              No PYQ documents found in {activeWorkspace.subject || 'this workspace'}
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 max-w-md mx-auto leading-relaxed">
              Upload past-year examination PDFs for <strong className="text-slate-900 dark:text-white">{activeWorkspace.subject || 'your subject'}</strong> to unlock automated study priorities and question analysis.
            </p>
          </div>
          <button
            onClick={() => setActiveTab('workspace')}
            className="px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-lg shadow-purple-500/20 inline-flex items-center space-x-2 transition-all"
          >
            <Upload className="w-4 h-4" />
            <span>Upload PYQs in Knowledge Base</span>
          </button>
        </div>
      ) : (
        /* WORKSPACE ALREADY CONTAINS PYQS - SHOW WORKSPACE OVERVIEW & STUDY PLAN (NO PRIMARY UPLOAD UI) */
        <>
          {/* Workspace Status & Uploaded Papers Overview Card */}
          <div className="saas-card p-5 space-y-4 border-slate-200 dark:border-purple-500/30 bg-gradient-to-r from-purple-100/70 via-indigo-50/70 to-slate-100 dark:from-[#0B1020] dark:via-[#0D1326] dark:to-[#0B1020]">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 dark:border-[#1F2937] pb-4">
              <div>
                <div className="flex items-center space-x-2">
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-purple-500/20 text-purple-700 dark:text-purple-300 border border-purple-500/30 uppercase">
                    Active Workspace
                  </span>
                  <span className="text-xs font-semibold text-slate-800 dark:text-white">
                    {activeWorkspace.university || 'University'} / {activeWorkspace.subject || 'Subject'} ({activeWorkspace.semester || 'Sem'})
                  </span>
                </div>
                <h3 className="font-heading font-extrabold text-lg text-slate-900 dark:text-white mt-1">
                  Ingested PYQ Intelligence Summary
                </h3>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => loadPriority(true)}
                  className="px-3.5 py-2 rounded-xl bg-slate-100 dark:bg-[#111827] hover:bg-slate-200 dark:hover:bg-[#1F2937] border border-slate-200 dark:border-[#1F2937] text-purple-600 dark:text-purple-300 text-xs font-semibold flex items-center space-x-1.5 transition-colors"
                  title="Re-run recurrence analysis on ingested papers"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Refresh Analysis</span>
                </button>

                <button
                  onClick={() => setActiveTab('workspace')}
                  className="px-3.5 py-2 rounded-xl bg-slate-100 dark:bg-[#111827] hover:bg-slate-200 dark:hover:bg-[#1F2937] border border-slate-200 dark:border-[#1F2937] text-slate-700 dark:text-slate-300 text-xs font-semibold flex items-center space-x-1.5 transition-colors"
                  title="Add more past year papers to workspace"
                >
                  <PlusCircle className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" />
                  <span>Add More PYQs</span>
                </button>

                <button
                  onClick={() => setIsSelectorOpen(true)}
                  className="px-3.5 py-2 rounded-xl bg-slate-100 dark:bg-[#111827] hover:bg-slate-200 dark:hover:bg-[#1F2937] border border-slate-200 dark:border-[#1F2937] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white text-xs font-semibold flex items-center space-x-1.5 transition-colors"
                >
                  <Layers className="w-3.5 h-3.5" />
                  <span>Switch Workspace</span>
                </button>
              </div>
            </div>

            {/* Quick Stat Pills */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-white dark:bg-[#080B14] border border-slate-200 dark:border-[#1F2937] flex items-center space-x-3">
                <div className="p-2 rounded-lg bg-purple-500/10 text-purple-600 dark:text-purple-400">
                  <FileText className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-[10px] uppercase font-bold text-slate-500">PYQ Papers</div>
                  <div className="font-heading font-extrabold text-sm text-slate-900 dark:text-white">{totalPapers} papers</div>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-white dark:bg-[#080B14] border border-slate-200 dark:border-[#1F2937] flex items-center space-x-3">
                <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">
                  <CheckCircle2 className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-[10px] uppercase font-bold text-slate-500">Canonical Questions</div>
                  <div className="font-heading font-extrabold text-sm text-slate-900 dark:text-white">{totalQuestions} extracted</div>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-white dark:bg-[#080B14] border border-slate-200 dark:border-[#1F2937] flex items-center space-x-3">
                <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                  <Calendar className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-[10px] uppercase font-bold text-slate-500">Years Covered</div>
                  <div className="font-heading font-extrabold text-sm text-slate-900 dark:text-white">{yearsRange}</div>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-white dark:bg-[#080B14] border border-slate-200 dark:border-[#1F2937] flex items-center space-x-3">
                <div className={`p-2 rounded-lg ${hasSyllabus ? 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400' : 'bg-amber-500/10 text-amber-600 dark:text-amber-400'}`}>
                  <BookOpen className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-[10px] uppercase font-bold text-slate-500">Syllabus Index</div>
                  <div className={`font-heading font-extrabold text-sm ${hasSyllabus ? 'text-indigo-600 dark:text-indigo-300' : 'text-amber-600 dark:text-amber-300'}`}>
                    {hasSyllabus ? 'Available' : 'Not Uploaded'}
                  </div>
                </div>
              </div>
            </div>

            {/* Uploaded Papers Checklist */}
            <div className="pt-2">
              <div className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2 flex items-center justify-between">
                <span>Ingested Examination Papers ({papersList.length || pyqFiles.length})</span>
                <span className="text-[10px] text-purple-600 dark:text-purple-400 font-normal">ChromaDB Grounded</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
                {papersList.length > 0 ? (
                  papersList.map((p, i) => (
                    <div key={i} className="p-2.5 rounded-lg bg-white dark:bg-[#080B14] border border-slate-200 dark:border-slate-800 flex items-center justify-between text-xs">
                      <div className="flex items-center space-x-2 truncate">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
                        <span className="font-semibold text-slate-900 dark:text-white truncate">
                          {p.exam_year ? `${p.exam_year} ${str(p.exam_session).split('/')[0]}` : p.source_file}
                        </span>
                      </div>
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded font-mono">
                        {p.valid_questions} Qs
                      </span>
                    </div>
                  ))
                ) : (
                  pyqFiles.map((f, i) => (
                    <div key={i} className="p-2.5 rounded-lg bg-white dark:bg-[#080B14] border border-slate-200 dark:border-slate-800 flex items-center justify-between text-xs">
                      <div className="flex items-center space-x-2 truncate">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
                        <span className="font-semibold text-slate-900 dark:text-white truncate">{f.name || f.id}</span>
                      </div>
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded font-mono">
                        {f.status || 'VERIFIED'}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* SYLLABUS STATUS BANNER */}
          {!hasSyllabus && (
            <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-700 dark:text-amber-200 text-xs flex items-center justify-between gap-3">
              <div className="flex items-center space-x-2.5">
                <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 flex-shrink-0" />
                <span>
                  <strong>PYQ analysis available.</strong> Upload a syllabus in Knowledge Base to enable module-wise topic mapping.
                </span>
              </div>
              <button
                onClick={() => setActiveTab('workspace')}
                className="px-3 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-700 dark:text-amber-200 text-xs font-semibold transition-colors flex-shrink-0"
              >
                Upload Syllabus
              </button>
            </div>
          )}

          {/* INCOMPLETE EXTRACTION SAFETY STATE */}
          {data?.extraction_incomplete && (
            <div className="saas-card p-6 text-center space-y-3 border-rose-500/30 bg-rose-500/5">
              <div className="flex items-center justify-center space-x-2 text-rose-400">
                <ShieldAlert className="w-5 h-5" />
                <h4 className="font-heading font-bold text-sm">Question Extraction Quality Notice</h4>
              </div>
              <p className="text-xs text-slate-300 max-w-lg mx-auto">
                {data?.prediction_notice || "Priority analysis limited because question extraction is incomplete. Valid grounded questions remain visible with appropriate confidence."}
              </p>
            </div>
          )}

          {/* Top Summary Stat Cards Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="saas-card p-4 space-y-1 border-rose-500/30">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>High Priority Topics</span>
                <TrendingUp className="w-4 h-4 text-rose-400" />
              </div>
              <div className="font-heading font-extrabold text-2xl text-white">
                {summary.high_priority_topics_count || 0}
              </div>
              <div className="text-[10px] text-rose-300 font-medium">Strong historical recurrence</div>
            </div>

            <div className="saas-card p-4 space-y-1 border-purple-500/30">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Repeated Questions</span>
                <Repeat className="w-4 h-4 text-purple-400" />
              </div>
              <div className="font-heading font-extrabold text-2xl text-white">
                {summary.repeated_questions_count || 0}
              </div>
              <div className="text-[10px] text-purple-300 font-medium">Exact & semantic repeat groups</div>
            </div>

            <div className="saas-card p-4 space-y-1 border-indigo-500/30">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Most Repeated Concept</span>
                <Sparkles className="w-4 h-4 text-indigo-400" />
              </div>
              <div className="font-heading font-bold text-sm text-white truncate">
                {summary.most_repeated_topic || 'N/A'}
              </div>
              <div className="text-[10px] text-indigo-300 font-medium">Top Priority Score #1</div>
            </div>

            <div className="saas-card p-4 space-y-1 border-emerald-500/30">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Years Analyzed</span>
                <Calendar className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="font-heading font-extrabold text-2xl text-white">
                {yearsRange}
              </div>
              <div className="text-[10px] text-emerald-300 font-medium">
                {(yearsCovered || []).join(', ')}
              </div>
            </div>
          </div>

          <div className="saas-card p-4 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div>
              <div className="text-[10px] uppercase text-slate-500 font-bold">Unique concepts</div>
              <div className="font-heading font-extrabold text-white">{summary.unique_concepts || topicPriorities.length || 0}</div>
            </div>
            <div>
              <div className="text-[10px] uppercase text-slate-500 font-bold">Exact repeats</div>
              <div className="font-heading font-extrabold text-white">{data?.exact_repeat_count || summary.exact_repeats || 0}</div>
            </div>
            <div>
              <div className="text-[10px] uppercase text-slate-500 font-bold">Semantic repeats</div>
              <div className="font-heading font-extrabold text-white">{data?.semantic_repeat_count || summary.semantic_repeats || 0}</div>
            </div>
            <div>
              <div className="text-[10px] uppercase text-slate-500 font-bold">Related topics</div>
              <div className="font-heading font-extrabold text-white">{(data?.related_topics || []).length || summary.related_topics || 0}</div>
            </div>
          </div>

          {/* Navigation Sub-Tabs */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 dark:border-[#1F2937] pb-3">
            <div className="flex space-x-2">
              <button
                onClick={() => setActiveSubTab('study-first')}
                className={`px-4 py-2 rounded-xl text-xs font-semibold border transition-all flex items-center space-x-2 ${
                  activeSubTab === 'study-first'
                    ? 'bg-purple-600 border-purple-500 text-white shadow-md glow-purple'
                    : 'bg-white dark:bg-[#0B1020] border-slate-200 dark:border-[#1F2937] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                <GraduationCap className="w-3.5 h-3.5" />
                <span>Recommended Study Order ("Study First")</span>
              </button>

              <button
                onClick={() => setActiveSubTab('repeated-questions')}
                className={`px-4 py-2 rounded-xl text-xs font-semibold border transition-all flex items-center space-x-2 ${
                  activeSubTab === 'repeated-questions'
                    ? 'bg-purple-600 border-purple-500 text-white shadow-md glow-purple'
                    : 'bg-white dark:bg-[#0B1020] border-slate-200 dark:border-[#1F2937] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                <Repeat className="w-3.5 h-3.5" />
                <span>High-Priority Repeated Questions ({questionPriorities.length})</span>
              </button>

              <button
                onClick={() => setActiveSubTab('recurring-topics')}
                className={`px-4 py-2 rounded-xl text-xs font-semibold border transition-all flex items-center space-x-2 ${
                  activeSubTab === 'recurring-topics'
                    ? 'bg-purple-600 border-purple-500 text-white shadow-md glow-purple'
                    : 'bg-white dark:bg-[#0B1020] border-slate-200 dark:border-[#1F2937] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                <Award className="w-3.5 h-3.5" />
                <span>High-Priority Recurring Topics ({topicPriorities.length})</span>
              </button>

              <button
                onClick={() => setActiveSubTab('related-topics')}
                className={`px-4 py-2 rounded-xl text-xs font-semibold border transition-all flex items-center space-x-2 ${
                  activeSubTab === 'related-topics'
                    ? 'bg-purple-600 border-purple-500 text-white shadow-md glow-purple'
                    : 'bg-white dark:bg-[#0B1020] border-slate-200 dark:border-[#1F2937] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                <BookOpen className="w-3.5 h-3.5" />
                <span>Related Topics ({(data?.related_topics || []).length})</span>
              </button>

              <button
                onClick={() => setActiveSubTab('modules')}
                className={`px-4 py-2 rounded-xl text-xs font-semibold border transition-all flex items-center space-x-2 ${
                  activeSubTab === 'modules'
                    ? 'bg-purple-600 border-purple-500 text-white shadow-md glow-purple'
                    : 'bg-white dark:bg-[#0B1020] border-slate-200 dark:border-[#1F2937] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Module-wise Priority</span>
              </button>
            </div>

            {data?.prediction_notice && (
              <span className="text-[11px] text-slate-400 italic">
                ℹ️ {data.prediction_notice}
              </span>
            )}
          </div>

          {/* TAB 1: RECOMMENDED STUDY ORDER ("STUDY FIRST") */}
          {activeSubTab === 'study-first' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-heading font-bold text-sm text-white uppercase tracking-wider">
                    Targeted Study Order for {activeWorkspace.subject || 'Uploaded Papers'}
                  </h3>
                  <p className="text-xs text-slate-400">
                    Ranked dynamically using historical frequency, distinct exam years, exact/semantic repeats, marks, and recency.
                  </p>
                </div>
              </div>

              {!recommendedPlan.length ? (
                <div className="saas-card p-8 text-center text-slate-400 text-xs">
                  No priority items generated.
                </div>
              ) : (
                <div className="space-y-6">
                  {['STUDY_FIRST', 'STUDY_NEXT', 'STUDY_AFTER', 'OPTIONAL'].map((band) => {
                    const items = recommendedPlan.filter((it) => (it.study_band || (it.rank <= 3 ? 'STUDY_FIRST' : 'STUDY_NEXT')) === band);
                    if (!items.length) return null;
                    const label = items[0].study_order_label || band.replace('_', ' ');
                    return (
                      <div key={band} className="space-y-3">
                        <h4 className="text-[11px] font-extrabold uppercase tracking-wider text-purple-300">{label}</h4>
                        {items.map((item, idx) => (
                          <PriorityCard
                            key={`${band}-${idx}`}
                            rank={item.rank || idx + 1}
                            item={item}
                            onWhy={(topic) => setSelectedWhyTopic(topic)}
                            onViewEvidence={(topic) => setSelectedEvidenceItem(topic)}
                            onStudy={(topicName) => handleStudyTopic(topicName)}
                          />
                        ))}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* TAB 2: HIGH-PRIORITY REPEATED QUESTIONS */}
          {activeSubTab === 'repeated-questions' && (
            <div className="space-y-4">
              <div>
                <h3 className="font-heading font-bold text-sm text-white uppercase tracking-wider">
                  Repeated Exam Questions
                </h3>
                <p className="text-xs text-slate-400">
                  Questions that have appeared exact or paraphrased across past exam papers.
                </p>
              </div>

              {!questionPriorities.length ? (
                <div className="saas-card p-8 text-center text-slate-400 text-xs">
                  No exact or semantic repeated question groups detected yet.
                </div>
              ) : (
                <div className="space-y-3">
                  {questionPriorities.map((q, idx) => (
                    <PriorityCard
                      key={idx}
                      rank={idx + 1}
                      item={{
                        ...q,
                        title: q.question_title || q.sample_text,
                        unit: q.syllabus_unit,
                      }}
                      onWhy={(topic) => setSelectedWhyTopic(topic)}
                      onViewEvidence={(topic) => setSelectedEvidenceItem(topic)}
                      onStudy={(txt) => handleStudyTopic(txt)}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* TAB 3: HIGH-PRIORITY RECURRING TOPICS */}
          {activeSubTab === 'recurring-topics' && (
            <div className="space-y-4">
              <div>
                <h3 className="font-heading font-bold text-sm text-white uppercase tracking-wider">
                  Recurring Academic Topic Clusters
                </h3>
                <p className="text-xs text-slate-400">
                  Broader topic concepts normalized generically without subject-specific hardcoding.
                </p>
              </div>

              {!topicPriorities.length ? (
                <div className="saas-card p-8 text-center text-slate-400 text-xs">
                  No topic clusters generated.
                </div>
              ) : (
                <div className="space-y-3">
                  {topicPriorities.map((t, idx) => (
                    <PriorityCard
                      key={idx}
                      rank={idx + 1}
                      item={t}
                      onWhy={(topic) => setSelectedWhyTopic(topic)}
                      onViewEvidence={(topic) => setSelectedEvidenceItem(topic)}
                      onStudy={(topicName) => handleStudyTopic(topicName)}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {activeSubTab === 'related-topics' && (
            <div className="space-y-4">
              <div>
                <h3 className="font-heading font-bold text-sm text-white uppercase tracking-wider">Related Topics</h3>
                <p className="text-xs text-slate-400">Shared concepts that are not exact or semantic repeats. Original source questions stay visible.</p>
              </div>
              {!(data?.related_topics || []).length ? (
                <div className="saas-card p-8 text-center text-slate-400 text-xs">No related-topic groups met the conservative threshold.</div>
              ) : (
                <div className="space-y-3">
                  {(data.related_topics || []).map((r, idx) => {
                    const members = (r.members && r.members.length)
                      ? r.members
                      : [r.q1, r.q2].filter(Boolean);
                    return (
                    <div key={idx} className="saas-card p-4 space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-extrabold px-2 py-0.5 rounded border bg-indigo-500/10 text-indigo-300 border-indigo-500/30">RELATED</span>
                        <span className="font-semibold text-white text-sm">{r.topic}</span>
                        <span className="text-[11px] text-slate-400">{members.length} questions · confidence {r.confidence ?? '—'}</span>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px] text-slate-300">
                        {members.map((m, mi) => (
                          <div key={mi} className="p-2 rounded bg-white dark:bg-[#080B14] border border-slate-200 dark:border-slate-800">
                            <div className="text-purple-600 dark:text-purple-300 font-semibold mb-1">{m?.source_ref}</div>
                            {m?.text}
                          </div>
                        ))}
                      </div>
                      <p className="text-[11px] text-slate-400 italic">Why grouped: {r.why_grouped || r.reason}</p>
                    </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {activeSubTab === 'modules' && (
            <div className="space-y-4">
              <div>
                <h3 className="font-heading font-bold text-sm text-white uppercase tracking-wider">Module-wise Priority</h3>
                <p className="text-xs text-slate-400">Mapped dynamically from the uploaded syllabus. Uncertain mappings are labelled, not invented.</p>
              </div>
              {!(data?.module_wise_priority || []).length ? (
                <div className="saas-card p-8 text-center text-slate-400 text-xs">No module mapping available yet. Upload a syllabus to enable this view.</div>
              ) : (
                <div className="space-y-3">
                  {(data.module_wise_priority || []).map((m, idx) => (
                    <div key={idx} className="saas-card p-4 space-y-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-heading font-bold text-white">{m.module}</span>
                        <span className="text-xs text-slate-400">Priority {m.priority} / 100</span>
                        {m.mapping_uncertain && (
                          <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-amber-500/10 text-amber-300 border-amber-500/30">Syllabus mapping uncertain</span>
                        )}
                      </div>
                      <div className="text-[11px] text-slate-300">Repeated concepts: {(m.repeated_concepts || []).join(' · ') || '—'}</div>
                      {(m.important_questions || []).length > 0 && (
                        <div className="text-[11px] text-slate-400 space-y-1">
                          {(m.important_questions || []).slice(0, 3).map((q, i) => (
                            <p key={i}>“{q}”</p>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}

    </div>
  );
}

// Helper safely stringifies session
function str(val) {
  if (!val) return '';
  return String(val);
}
