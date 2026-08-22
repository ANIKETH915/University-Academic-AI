import React from 'react';

export default function StatCard({ label, value, subtext, icon: Icon, accentColor = 'purple' }) {
  const accentStyles = {
    purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20'
  };

  return (
    <div className="saas-card p-5 flex flex-col justify-between space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
          {label}
        </span>
        {Icon && (
          <div className={`p-2 rounded-xl border ${accentStyles[accentColor] || accentStyles.purple}`}>
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>

      <div>
        <div className="font-heading font-extrabold text-2xl sm:text-3xl text-white tracking-tight">
          {value}
        </div>
        {subtext && (
          <div className="text-[11px] font-medium text-slate-400 mt-1">
            {subtext}
          </div>
        )}
      </div>
    </div>
  );
}
