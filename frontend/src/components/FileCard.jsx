import React from 'react';
import { FileText, CheckCircle2, MoreVertical, Trash2 } from 'lucide-react';

export default function FileCard({ filename, size, metadata, status = 'VERIFIED', onRemove }) {
  return (
    <div className="p-4 rounded-xl bg-[#0B1020] border border-[#1F2937] hover:border-[#374151] transition-all flex items-center justify-between gap-3 group">
      <div className="flex items-center space-x-3 truncate">
        <div className="w-9 h-9 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 flex items-center justify-center flex-shrink-0">
          <FileText className="w-4 h-4" />
        </div>

        <div className="truncate">
          <h5 className="font-heading font-bold text-xs text-slate-200 truncate group-hover:text-white transition-colors">
            {filename}
          </h5>
          <div className="text-[11px] text-slate-400 mt-0.5 flex items-center space-x-2">
            <span>{size}</span>
            {metadata && (
              <>
                <span>•</span>
                <span className="text-purple-300 font-medium">{metadata}</span>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="flex items-center space-x-2 flex-shrink-0">
        <span className="text-[10px] font-extrabold px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center space-x-1">
          <CheckCircle2 className="w-3 h-3" />
          <span>{status}</span>
        </span>
        {onRemove && (
          <button
            onClick={onRemove}
            className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
            title="Remove document"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}
