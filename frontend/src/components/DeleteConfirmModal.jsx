import React from 'react';
import { AlertTriangle, Loader2, X } from 'lucide-react';

export default function DeleteConfirmModal({
  isOpen,
  filename,
  onConfirm,
  onCancel,
  isDeleting = false
}) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xs animate-in fade-in duration-150">
      <div 
        className="w-full max-w-md bg-white dark:bg-[#111827] border border-slate-200 dark:border-[#1F2937] rounded-2xl p-6 shadow-2xl space-y-5 text-slate-900 dark:text-slate-100 relative"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Icon */}
        <button
          onClick={onCancel}
          disabled={isDeleting}
          className="absolute top-4 right-4 p-1 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-white transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Modal Header */}
        <div className="flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-500 flex items-center justify-center flex-shrink-0">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-heading font-extrabold text-lg text-slate-900 dark:text-white tracking-tight">
              Delete document?
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Destructive action — requires confirmation
            </p>
          </div>
        </div>

        {/* Body Text */}
        <div className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed bg-slate-50 dark:bg-[#080B14] p-3.5 rounded-xl border border-slate-200 dark:border-[#1F2937]">
          Are you sure you want to delete <span className="font-bold text-slate-900 dark:text-white underline decoration-rose-500/40">{filename}</span>?
          <br />
          This will remove the uploaded document and its associated processed data.
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end space-x-3 pt-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={isDeleting}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-all border border-slate-200 dark:border-slate-700"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isDeleting}
            className="px-4 py-2 rounded-xl text-xs font-bold bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/20 transition-all flex items-center space-x-2 border border-rose-500/50 disabled:opacity-50"
          >
            {isDeleting ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Deleting...</span>
              </>
            ) : (
              <span>Delete</span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
