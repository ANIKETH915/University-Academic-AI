export function classifyQueryIntent(queryText) {
  if (!queryText || typeof queryText !== 'string') return 'GENERAL_QA';
  const q = queryText.toLowerCase().trim();

  // PYQ Analysis Intent Signals
  const pyqSignals = [
    'repeatedly asked', 'repeated question', 'recurring question', 'most frequent',
    'which questions appeared', 'what questions appear', 'pyq analysis',
    'past exam questions', 'questions repeated', 'frequently asked', 'recurrence',
    'questions in previous papers', 'topics repeated', 'questions keep coming'
  ];

  if (pyqSignals.some(signal => q.includes(signal))) {
    return 'PYQ_ANALYSIS';
  }

  // Study Priority Intent Signals
  const prioritySignals = [
    'study first', 'should i study', 'study priority', 'most important topic',
    'highest priority', 'what to focus on', 'focus study', 'what to prepare first',
    'top topics for exam', 'how should i prepare', 'study plan', 'what should i focus'
  ];

  if (prioritySignals.some(signal => q.includes(signal))) {
    return 'STUDY_PRIORITY';
  }

  return 'GENERAL_QA';
}
