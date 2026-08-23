import React from 'react';
import { FileText, CheckCircle2, Lock, Trash2 } from 'lucide-react';

export default function FileCard({ filename, size, metadata, status = 'Ready', buildStarted = false, isPending = false, onRemove }) {
  const upperStatus = String(status).toUpperCase();
  const isPendingDoc = isPending || (upperStatus === 'PENDING' && !buildStarted);

  const getBadgeStyle = () => {
    if (upperStatus.includes('PROCESS') || upperStatus.includes('INGEST')) {
      return 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20';
    }
    if (upperStatus.includes('FAIL') || upperStatus.includes('ERR')) {
      return 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20';
    }
    if (isPendingDoc) {
      return 'bg-purple-500/10 text-purple-600 dark:text-purple-300 border-purple-500/20';
    }
    return 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20';
  };

  return (
    <div className="p-4 rounded-xl bg-white dark:bg-[#0B1020] border border-slate-200 dark:border-[#1F2937] hover:border-slate-300 dark:hover:border-[#374151] transition-all flex items-center justify-between gap-3 group shadow-xs">
      <div className="flex items-center space-x-3 truncate">
        <div className="w-9 h-9 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-600 dark:text-purple-400 flex items-center justify-center flex-shrink-0">
          <FileText className="w-4 h-4" />
        </div>

        <div className="truncate">
          <h5 className="font-heading font-bold text-xs text-slate-800 dark:text-slate-200 truncate group-hover:text-slate-900 dark:group-hover:text-white transition-colors">
            {filename}
          </h5>
          <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 flex items-center space-x-2">
            <span>{size}</span>
            {metadata && (
              <>
                <span>•</span>
                <span className="text-purple-600 dark:text-purple-300 font-medium">{metadata}</span>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="flex items-center space-x-2 flex-shrink-0">
        <span className={`text-[10px] font-extrabold px-2.5 py-1 rounded-md border flex items-center space-x-1 ${getBadgeStyle()}`}>
          {isPendingDoc ? (
            <span>PENDING BUILD</span>
          ) : (
            <>
              <Lock className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
              <span>{upperStatus === 'VERIFIED' ? 'READY (LOCKED)' : `${status} (LOCKED)`}</span>
            </>
          )}
        </span>

        {/* Delete is ONLY available for pending/unbuilt documents */}
        {isPendingDoc && onRemove && (
          <button
            onClick={onRemove}
            className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 hover:bg-rose-500/10 transition-colors border border-transparent hover:border-rose-500/20"
            title="Delete pending document (before build)"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}
