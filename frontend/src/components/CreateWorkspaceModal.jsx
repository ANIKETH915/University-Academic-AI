import React, { useState } from 'react';
import { useWorkspace } from '../context/WorkspaceContext';
import { Plus, X, CheckCircle, AlertCircle } from 'lucide-react';

export default function CreateWorkspaceModal() {
  const { isCreateOpen, setIsCreateOpen, createWorkspace } = useWorkspace();

  const [formData, setFormData] = useState({
    university: 'Anna University',
    branch: 'Computer Science and Engineering',
    semester: 'Semester 5',
    subject: 'Computer Networks',
    subjectCode: 'CS3591'
  });

  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState('');

  if (!isCreateOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (isCreating) return;

    if (typeof createWorkspace !== 'function') {
      console.error('[WORKSPACE] createWorkspace is unavailable');
      setError('Workspace creation is currently unavailable.');
      return;
    }

    if (!formData.university.trim() || !formData.subject.trim()) {
      setError('University and Subject Name are required.');
      return;
    }

    try {
      setIsCreating(true);
      setError('');
      const created = await createWorkspace({
        university: formData.university.trim(),
        program: formData.branch.trim(),
        branch: formData.branch.trim(),
        semester: formData.semester,
        subject: formData.subject.trim(),
        subjectCode: formData.subjectCode.trim()
      });

      if (!created?.id) {
        setError('Backend did not return a workspace ID. Creation failed.');
        return;
      }

      console.log('[WORKSPACE CREATE] backend_id=', created.id, 'subject=', created.subject);
      console.log('[WORKSPACE ACTIVE]', created.id);

      setIsCreateOpen(false);
    } catch (err) {
      console.error('[WORKSPACE SUBMIT ERROR]', err);
      setError('Failed to create workspace. Please try again.');
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-4">
      <div className="saas-card w-full max-w-lg p-6 space-y-5 animate-in fade-in zoom-in-95 duration-150">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#1F2937] pb-4">
          <div className="flex items-center space-x-2 text-purple-400">
            <Plus className="w-5 h-5" />
            <div>
              <h3 className="font-heading font-extrabold text-base text-white">CREATE ACADEMIC WORKSPACE</h3>
              <p className="text-xs text-slate-400">Define a dedicated knowledge space for any course.</p>
            </div>
          </div>
          <button
            onClick={() => setIsCreateOpen(false)}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-[#1F2937] transition-colors"
            disabled={isCreating}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {error && (
          <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        {/* Wizard Form */}
        <form onSubmit={handleSubmit} className="space-y-4 text-xs">
          
          {/* Step 1: University */}
          <div>
            <label className="block text-[11px] font-semibold text-slate-400 mb-1 uppercase tracking-wider">
              Step 1: University / Institute Name
            </label>
            <input
              type="text"
              required
              value={formData.university}
              onChange={(e) => setFormData({ ...formData, university: e.target.value })}
              placeholder="e.g. Anna University, IGNOU, GTU, VTU"
              className="w-full bg-[#0B1020] border border-[#1F2937] rounded-xl px-3.5 py-2.5 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30"
            />
          </div>

          {/* Step 2: Branch / Program */}
          <div>
            <label className="block text-[11px] font-semibold text-slate-400 mb-1 uppercase tracking-wider">
              Step 2: Branch / Program Stream
            </label>
            <input
              type="text"
              required
              value={formData.branch}
              onChange={(e) => setFormData({ ...formData, branch: e.target.value })}
              placeholder="e.g. Computer Science and Engineering, B.Com, Information Technology"
              className="w-full bg-[#0B1020] border border-[#1F2937] rounded-xl px-3.5 py-2.5 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30"
            />
          </div>

          {/* Step 3 & Step 4: Semester & Subject */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block text-[11px] font-semibold text-slate-400 mb-1 uppercase tracking-wider">
                Step 3: Semester
              </label>
              <select
                value={formData.semester}
                onChange={(e) => setFormData({ ...formData, semester: e.target.value })}
                className="w-full bg-[#0B1020] border border-[#1F2937] rounded-xl px-3.5 py-2.5 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30"
              >
                {['Semester 1', 'Semester 2', 'Semester 3', 'Semester 4', 'Semester 5', 'Semester 6', 'Semester 7', 'Semester 8'].map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-[11px] font-semibold text-slate-400 mb-1 uppercase tracking-wider">
                Step 4: Subject Name
              </label>
              <input
                type="text"
                required
                value={formData.subject}
                onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                placeholder="e.g. Financial Accounting, Computer Networks, DBMS"
                className="w-full bg-[#0B1020] border border-[#1F2937] rounded-xl px-3.5 py-2.5 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30"
              />
            </div>
          </div>

          {/* Step 5: Subject Code */}
          <div>
            <label className="block text-[11px] font-semibold text-slate-400 mb-1 uppercase tracking-wider">
              Step 5: Subject Code (Optional)
            </label>
            <input
              type="text"
              value={formData.subjectCode}
              onChange={(e) => setFormData({ ...formData, subjectCode: e.target.value })}
              placeholder="e.g. BCOC-131, CS3591"
              className="w-full bg-[#0B1020] border border-[#1F2937] rounded-xl px-3.5 py-2.5 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30"
            />
          </div>

          {/* Buttons */}
          <div className="pt-3 border-t border-[#1F2937] flex items-center justify-end space-x-3">
            <button
              type="button"
              onClick={() => setIsCreateOpen(false)}
              className="px-4 py-2 rounded-xl bg-[#0B1020] border border-[#1F2937] text-slate-300 hover:text-white text-xs font-semibold"
              disabled={isCreating}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isCreating}
              className="px-5 py-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-md shadow-purple-500/20 flex items-center space-x-1.5 disabled:opacity-50"
            >
              <CheckCircle className="w-4 h-4" />
              <span>{isCreating ? 'Creating Workspace...' : 'Create Workspace'}</span>
            </button>
          </div>

        </form>

      </div>
    </div>
  );
}
