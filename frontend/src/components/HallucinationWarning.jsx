import React from 'react';
import { AlertTriangle, ShieldCheck } from 'lucide-react';

export default function HallucinationWarning({ topicName }) {
  return (
    <div className="bg-amber-950/40 border border-amber-500/30 rounded-xl p-5 mb-6 text-amber-200 shadow-lg">
      <div className="flex items-start space-x-3">
        <div className="p-2 rounded-lg bg-amber-500/20 text-amber-400 mt-0.5">
          <AlertTriangle className="w-5 h-5" />
        </div>
        <div>
          <h4 className="font-heading font-bold text-amber-300 text-sm flex items-center space-x-2">
            <span>⚠️ Topic Not Found in Your Documents</span>
          </h4>
          <p className="text-xs text-amber-200/90 mt-1 leading-relaxed">
            The topic <strong className="text-amber-100 font-semibold">"{topicName}"</strong> was not found in the documents uploaded to this academic workspace.
          </p>
          <div className="mt-3 flex items-center space-x-2 text-[11px] text-amber-400/90 font-medium bg-amber-900/30 px-3 py-1.5 rounded-md border border-amber-500/20">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Grounded Hallucination Guard: Answers are restricted strictly to your uploaded documents.</span>
          </div>
        </div>
      </div>
    </div>
  );
}
