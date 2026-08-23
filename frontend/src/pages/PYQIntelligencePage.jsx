import React, { useState, useEffect } from 'react';
import { useWorkspace } from '../context/WorkspaceContext';
import { fetchPYQAnalysis, fetchPYQSourceQuestions } from '../api/academicApi';
import PageHeader from '../components/PageHeader';
import PriorityCard from '../components/PriorityCard';
import WhyPriorityModal from '../components/WhyPriorityModal';
import {
  Flame, Loader2, Layers, Upload, Database, AlertCircle, FileText,
  GitCompare, Info, Target, Repeat, Link2, X
} from 'lucide-react';

export default function PYQIntelligencePage({ setActiveTab, setSelectedTopicForAsk }) {
  const { activeWorkspace, setIsSelectorOpen } = useWorkspace();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedWhyTopic, setSelectedWhyTopic] = useState(null);
  const [showSourceModal, setShowSourceModal] = useState(false);
  const [sourceQuestions, setSourceQuestions] = useState([]);
  const [loadingSources, setLoadingSources] = useState(false);

  const isInvalidWorkspace = !activeWorkspace || !activeWorkspace.id || activeWorkspace.id === '/' || activeWorkspace.id === 'undefined' || activeWorkspace.id === 'null';
  // Backend vectors are source of truth — do NOT gate on local pyqFiles alone
  const localPyqHint = activeWorkspace && activeWorkspace.pyqFiles && activeWorkspace.pyqFiles.length > 0;

  useEffect(() => {
    async function loadData() {
      setData(null);
      if (isInvalidWorkspace) {
        setLoading(false);
        return;
      }
      setLoading(true);
      console.log('[ANALYSIS] workspace_id=', activeWorkspace.id);
      const res = await fetchPYQAnalysis(
        activeWorkspace.subject || 'Subject',
        activeWorkspace.semester || 'Semester',
        true, // always query backend for this workspace
        activeWorkspace.id
      );
      console.log(
        '[ANALYSIS] workspace_id=', res?.workspace_id || activeWorkspace.id,
        'questions=', res?.total_valid_questions ?? res?.total_questions_analyzed,
        'papers=', res?.total_papers
      );
      setData(res);
      setLoading(false);
    }
    loadData();
  }, [activeWorkspace?.id, activeWorkspace?.subject, activeWorkspace?.semester, activeWorkspace?.pyqFiles?.length]);

  const handleStudyTopic = (topicName) => {
    if (setSelectedTopicForAsk) setSelectedTopicForAsk(topicName);
    setActiveTab('ask');
  };

  const openSourceQuestions = async () => {
    setShowSourceModal(true);
    setLoadingSources(true);
    try {
      const res = await fetchPYQSourceQuestions(activeWorkspace.id);
      setSourceQuestions(res?.accepted_questions || []);
    } catch {
      setSourceQuestions([]);
    }
    setLoadingSources(false);
  };

  const isSinglePaper = data?.single_paper_mode || (data?.total_papers === 1);
  const yearsLabel = (data?.years_covered || []).join(', ') || '—';

  return (
    <div className="space-y-6">
      <PageHeader
        breadcrumb="PYQ Intelligence"
        title="PYQ Intelligence"
        subtitle={`Question-level historical intelligence for ${activeWorkspace?.subject || 'your subject'} — exact repeats, semantic repeats, and related topics kept separate.`}
        icon={Flame}
        badge={isSinglePaper ? 'Single Paper Diagnostic' : 'Multi-Year Recurrence Engine'}
      />

      <div className="p-4 rounded-xl bg-[#0B1020] border border-[#1F2937] text-slate-300 text-xs flex items-center justify-between gap-3">
        <div>
          <span className="font-semibold text-white">Analysis Scope:</span>{' '}
          Active workspace only — <strong className="text-purple-300">{activeWorkspace?.university || 'Workspace'} / {activeWorkspace?.subject || 'Subject'} ({activeWorkspace?.semester || 'Sem'})</strong>.
        </div>
        <button
          onClick={() => setIsSelectorOpen(true)}
          className="px-3 py-1.5 rounded-lg bg-[#111827] hover:bg-[#1F2937] border border-[#1F2937] text-purple-300 text-xs font-semibold flex items-center space-x-1 transition-colors flex-shrink-0"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Switch Workspace</span>
        </button>
      </div>

      {isInvalidWorkspace ? (
        <div className="saas-card p-10 text-center space-y-4 max-w-2xl mx-auto my-6">
          <div className="w-16 h-16 rounded-3xl bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center justify-center mx-auto">
            <AlertCircle className="w-8 h-8" />
          </div>
          <div className="space-y-1">
            <h3 className="font-heading font-extrabold text-lg text-white">No active workspace selected</h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
              Create or select an active academic workspace to analyze examination questions.
            </p>
          </div>
          <button onClick={() => setIsSelectorOpen(true)} className="px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-lg shadow-purple-500/20 inline-flex items-center space-x-2">
            <Layers className="w-4 h-4" /><span>Select Workspace</span>
          </button>
        </div>
      ) : loading ? (
        <div className="p-12 text-center text-slate-400 saas-card">
          <Loader2 className="w-8 h-8 animate-spin mx-auto text-purple-400 mb-2" />
          <p className="text-xs">Computing question-level intelligence for {activeWorkspace.subject || 'uploaded papers'}...</p>
        </div>
      ) : data?.unavailable ? (
        <div className="saas-card p-10 text-center space-y-4 max-w-2xl mx-auto my-6">
          <div className="w-16 h-16 rounded-3xl bg-rose-500/10 border border-rose-500/20 text-rose-400 flex items-center justify-center mx-auto">
            <AlertCircle className="w-8 h-8" />
          </div>
          <div className="space-y-1">
            <h3 className="font-heading font-extrabold text-lg text-white">Intelligence could not be loaded</h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
              The backend rejected or could not serve this request, so no analysis is shown.
              This is not the same as an empty workspace.
            </p>
            <p className="text-xs text-rose-300 font-mono break-words max-w-md mx-auto pt-2">
              {data.error}{data.error_status ? ` (HTTP ${data.error_status})` : ''}
            </p>
          </div>
          <button onClick={() => setActiveTab('workspace')} className="px-6 py-3 rounded-xl bg-[#111827] hover:bg-[#1F2937] border border-[#1F2937] text-purple-300 font-semibold text-xs inline-flex items-center space-x-2">
            <Database className="w-4 h-4" /><span>Open Knowledge Base</span>
          </button>
        </div>
      ) : !(data?.total_valid_questions || data?.total_questions_analyzed || data?.total_papers) ? (
        <div className="saas-card p-10 text-center space-y-4 max-w-2xl mx-auto my-6">
          <div className="w-16 h-16 rounded-3xl bg-purple-500/10 border border-purple-500/20 text-purple-400 flex items-center justify-center mx-auto">
            <Database className="w-8 h-8" />
          </div>
          <div className="space-y-1">
            <h3 className="font-heading font-extrabold text-lg text-white">No previous-year papers in this workspace yet</h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
              Upload PYQ PDFs for <strong className="text-white">{activeWorkspace.subject || 'your subject'}</strong> to unlock recurrence intelligence.
              {localPyqHint ? ' (Local file list exists but backend returned no PYQ vectors — re-ingest may be required.)' : ''}
            </p>
          </div>
          <button onClick={() => setActiveTab('workspace')} className="px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-lg shadow-purple-500/20 inline-flex items-center space-x-2">
            <Upload className="w-4 h-4" /><span>Upload PYQs</span>
          </button>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Overall Statistics */}
          <div>
            <h3 className="font-heading font-bold text-xs text-white uppercase tracking-wider mb-3">Overall Statistics</h3>
            <div className="saas-card p-5 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-4 text-center">
              <Stat label="Questions analyzed" value={data?.total_valid_questions || data?.total_questions_analyzed || 0} color="text-purple-400" />
              <Stat label="Papers analyzed" value={data?.total_papers || 0} color="text-indigo-400" />
              <Stat label="Years covered" value={yearsLabel} color="text-blue-400" small />
              <Stat label="Unique intents" value={data?.unique_question_intents || 0} color="text-cyan-400" />
              <Stat label="Exact repeats" value={data?.exact_repeat_count || data?.exact_repeats?.length || 0} color="text-emerald-400" />
              <Stat label="Semantic repeats" value={data?.semantic_repeat_count || data?.semantic_repeats?.length || 0} color="text-amber-400" />
              <Stat label="Repeated topics" value={data?.topic_recurrence?.length || data?.unique_topic_clusters || 0} color="text-rose-400" />
            </div>
            <div className="mt-3 flex justify-end">
              <button
                onClick={openSourceQuestions}
                className="px-3 py-1.5 rounded-lg bg-[#111827] hover:bg-[#1F2937] border border-[#1F2937] text-slate-300 text-xs font-semibold inline-flex items-center gap-1.5"
              >
                <FileText className="w-3.5 h-3.5" />
                View Source Questions
              </button>
            </div>
          </div>

          {isSinglePaper && (
            <div className="p-4 rounded-xl bg-purple-500/10 border border-purple-500/30 text-purple-300 text-xs flex items-start gap-3">
              <Info className="w-5 h-5 flex-shrink-0 text-purple-400 mt-0.5" />
              <div>
                <h4 className="font-bold text-white text-sm">Single paper mode</h4>
                <p className="mt-1 leading-relaxed text-slate-300">
                  Historical multi-year prediction stays inactive until more papers are uploaded. Within-paper patterns and source validation still apply.
                </p>
              </div>
            </div>
          )}

          {/* Most Repeated Questions */}
          <Section title="Most Repeated Questions" icon={Repeat}>
            {(data?.most_repeated_questions || []).length === 0 ? (
              <Empty note="Not enough evidence of repeated questions across papers yet." />
            ) : (
              <div className="space-y-3">
                {data.most_repeated_questions.slice(0, 12).map((item, idx) => (
                  <div key={idx} className="p-4 rounded-xl bg-[#0B1020] border border-[#1F2937] space-y-2">
                    <div className="flex items-start justify-between gap-3">
                      <h4 className="font-semibold text-sm text-white">{idx + 1}. {item.title}</h4>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-purple-500/10 border border-purple-500/20 text-purple-300 uppercase">
                        {item.kind}
                      </span>
                    </div>
                    <div className="text-xs text-slate-400 grid grid-cols-2 sm:grid-cols-4 gap-2">
                      <span>Asked: <strong className="text-slate-200">{item.asked_count}</strong> times</span>
                      <span>Years: <strong className="text-slate-200">{(item.years || []).join(', ')}</strong></span>
                      <span>Exact: <strong className="text-slate-200">{item.exact_repeats || 0}</strong></span>
                      <span>Semantic: <strong className="text-slate-200">{item.semantic_repeats || 0}</strong></span>
                    </div>
                    <div className="text-[11px] text-slate-300">
                      <span className="text-slate-500 font-semibold uppercase tracking-wide">Sources:</span>{' '}
                      {(item.sources || []).join(' · ')}
                    </div>
                    {item.why_same && <p className="text-[11px] text-slate-400 italic">Why same: {item.why_same}</p>}
                  </div>
                ))}
              </div>
            )}
          </Section>

          {/* Exact Repeats */}
          <Section title="Exact Repeats" icon={GitCompare}>
            {(data?.exact_repeats || []).length === 0 ? (
              <Empty note="No exact normalized repeats detected." />
            ) : (
              <div className="space-y-3">
                {data.exact_repeats.map((g, idx) => (
                  <div key={idx} className="p-4 rounded-xl bg-[#0B1020] border border-[#1F2937] space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-extrabold px-2 py-0.5 rounded border bg-emerald-500/10 text-emerald-300 border-emerald-500/30">{g.group_type || 'EXACT'}</span>
                      <span className="text-[11px] text-slate-400">conf {g.confidence ?? 1}</span>
                    </div>
                    <p className="text-xs text-slate-200 leading-relaxed">{g.exact_text}</p>
                    {(g.original_questions || []).length > 0 && (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {g.original_questions.map((oq, i2) => (
                          <div key={i2} className="p-2 rounded bg-[#111827] border border-[#1F2937] text-[11px] text-slate-300">
                            <div className="text-purple-300 font-semibold mb-1">{oq.source_ref}</div>
                            {oq.text}
                          </div>
                        ))}
                      </div>
                    )}
                    <div className="text-[11px] text-slate-400">
                      Years: {(g.years || []).join(', ')} · {(g.source_refs || []).join(' · ')}
                    </div>
                    <p className="text-[11px] text-emerald-400/90">Why grouped: {g.why_grouped || g.reason || 'Normalized wording essentially identical'}</p>
                  </div>
                ))}
              </div>
            )}
          </Section>

          {/* Semantic Repeats */}
          <Section title="Semantic / Paraphrased Repeats" icon={Link2}>
            {(data?.semantic_repeats || []).length === 0 ? (
              <Empty note="No high-confidence semantic repeats yet — preferring no false positives." />
            ) : (
              <div className="space-y-3">
                {data.semantic_repeats.map((g, idx) => (
                  <div key={idx} className="p-4 rounded-xl bg-[#0B1020] border border-[#1F2937] space-y-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[10px] font-extrabold px-2 py-0.5 rounded border bg-amber-500/10 text-amber-300 border-amber-500/30">{g.group_type || 'SEMANTIC'}</span>
                      <h4 className="text-sm font-semibold text-white">{g.display_title}</h4>
                      <span className="text-[11px] text-slate-400">conf {g.confidence ?? '—'}</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {(g.original_questions || []).map((oq, i2) => (
                        <div key={i2} className="p-2 rounded bg-[#111827] border border-[#1F2937] text-[11px] text-slate-300">
                          <div className="text-purple-300 font-semibold mb-1">{oq.source_ref}</div>
                          {oq.text}
                        </div>
                      ))}
                    </div>
                    <p className="text-[11px] text-amber-300/90">Why they are considered the same: {g.why_same || g.reason}</p>
                  </div>
                ))}
              </div>
            )}
          </Section>

          {/* Related Topics */}
          <Section title="Related Topics (not repeats)" icon={Link2}>
            <p className="text-[11px] text-slate-500 mb-3">
              Broader than a repeat: shared concept, different academic ask. Original questions stay visible.
            </p>
            {(data?.related_topics || []).length === 0 ? (
              <Empty note="No related-topic pairs met the conservative threshold." />
            ) : (
              <div className="space-y-3">
                {data.related_topics.slice(0, 20).map((r, idx) => (
                  <div key={idx} className="p-4 rounded-xl bg-[#0B1020] border border-[#1F2937] space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[10px] font-extrabold px-2 py-0.5 rounded border bg-indigo-500/10 text-indigo-300 border-indigo-500/30">{r.group_type || 'RELATED'}</span>
                      <span className="font-semibold text-white text-sm">{r.topic}</span>
                      <span className="text-[11px] text-slate-400">sim {r.similarity ?? '—'} · conf {r.confidence ?? '—'}</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <div className="p-2 rounded bg-[#111827] border border-[#1F2937] text-[11px] text-slate-300">
                        <div className="text-purple-300 font-semibold mb-1">{r.q1?.source_ref}</div>
                        {r.q1?.text}
                      </div>
                      <div className="p-2 rounded bg-[#111827] border border-[#1F2937] text-[11px] text-slate-300">
                        <div className="text-purple-300 font-semibold mb-1">{r.q2?.source_ref}</div>
                        {r.q2?.text}
                      </div>
                    </div>
                    <p className="text-[11px] text-slate-400 italic">Why grouped: {r.why_grouped || r.reason}</p>
                  </div>
                ))}
              </div>
            )}
          </Section>

          {/* Topic Recurrence */}
          <Section title="Topic Recurrence" icon={Layers}>
            <p className="text-[11px] text-slate-500 mb-3">
              Broader than question recurrence — related questions on the same topic are not automatically marked as repeats.
            </p>
            {(data?.topic_recurrence || []).length === 0 ? (
              <Empty note="No topic recurrence evidence yet." />
            ) : (
              <div className="space-y-2">
                {(data.topic_recurrence || []).slice(0, 20).map((t, idx) => (
                  <div key={idx} className="flex flex-wrap items-center justify-between gap-2 p-3 rounded-lg bg-[#0B1020] border border-[#1F2937] text-xs">
                    <span className="font-semibold text-white">{t.topic}</span>
                    <span className="text-slate-400">Appeared in: <strong className="text-slate-200">{(t.years || []).join(', ')}</strong> ({t.appearances}×)</span>
                  </div>
                ))}
              </div>
            )}
          </Section>

          {/* Study Priority */}
          <Section title="Study Priority" icon={Target}>
            {(data?.study_priorities || data?.topics || []).length === 0 ? (
              <Empty note="Insufficient evidence for priority ranking." />
            ) : (
              <div className="space-y-3">
                {(data.study_priorities || data.topics || []).slice(0, 8).map((topic, idx) => (
                  <PriorityCard
                    key={idx}
                    rank={topic.rank || idx + 1}
                    topic={{
                      ...topic,
                      appearances_count: topic.appearances_count || topic.appearances,
                      years_appeared: topic.years_appeared || topic.years,
                    }}
                    onWhy={(t) => setSelectedWhyTopic(t)}
                    onStudy={(topicName) => handleStudyTopic(topicName)}
                  />
                ))}
              </div>
            )}
          </Section>
        </div>
      )}

      <WhyPriorityModal topic={selectedWhyTopic} onClose={() => setSelectedWhyTopic(null)} />

      {showSourceModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70">
          <div className="w-full max-w-3xl max-h-[80vh] overflow-hidden rounded-2xl bg-[#0B1020] border border-[#1F2937] shadow-2xl flex flex-col">
            <div className="flex items-center justify-between px-5 py-4 border-b border-[#1F2937]">
              <h3 className="font-heading font-bold text-sm text-white">Source Questions (valid canonical PYQs only)</h3>
              <button onClick={() => setShowSourceModal(false)} className="text-slate-400 hover:text-white"><X className="w-5 h-5" /></button>
            </div>
            <div className="overflow-y-auto p-5 space-y-3">
              {loadingSources ? (
                <div className="py-10 text-center text-slate-400 text-xs"><Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />Loading…</div>
              ) : sourceQuestions.length === 0 ? (
                <p className="text-xs text-slate-400 text-center py-8">No valid source questions.</p>
              ) : (
                sourceQuestions.map((q, idx) => (
                  <div key={idx} className="p-3 rounded-lg bg-[#111827] border border-[#1F2937]">
                    <div className="flex flex-wrap items-center gap-2 text-[11px] text-purple-300 font-semibold mb-1">
                      <span>{q.question_number || q.question_id}</span>
                      {q.marks != null && <span className="text-slate-500">· {q.marks}M</span>}
                      {q.year != null && <span className="text-slate-500">· {q.year}</span>}
                      {q.source_page != null && <span className="text-slate-500">· p.{q.source_page}</span>}
                    </div>
                    <p className="text-[10px] text-slate-500 mb-1 truncate">{q.source_file || q.source_ref}</p>
                    <p className="text-xs text-slate-200 leading-relaxed">{q.exact_text}</p>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, color, small }) {
  return (
    <div>
      <div className={`font-heading font-extrabold ${small ? 'text-sm' : 'text-2xl'} ${color}`}>{value}</div>
      <div className="text-[10px] text-slate-500 font-semibold uppercase mt-0.5">{label}</div>
    </div>
  );
}

function Section({ title, icon: Icon, children }) {
  return (
    <div className="saas-card p-6 space-y-4">
      <div className="flex items-center space-x-2 border-b border-[#1F2937] pb-3">
        <Icon className="w-5 h-5 text-purple-400" />
        <h3 className="font-heading font-bold text-sm text-white uppercase tracking-wider">{title}</h3>
      </div>
      {children}
    </div>
  );
}

function Empty({ note }) {
  return <div className="p-6 text-center text-slate-400 bg-[#0B1020] rounded-xl border border-[#1F2937] text-xs">{note}</div>;
}
