import React from 'react';
import { ChevronRight } from 'lucide-react';

export default function PageHeader({ breadcrumb = 'Dashboard', title, subtitle, icon: Icon, badge }) {
  return (
    <div className="saas-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div>
        <div className="flex items-center space-x-2 text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
          <span className="text-slate-500 dark:text-slate-400">University Academic AI</span>
          <ChevronRight className="w-3 h-3 text-slate-400 dark:text-slate-600" />
          <span className="text-purple-600 dark:text-purple-400 font-semibold">{breadcrumb}</span>
          {badge && (
            <span className="ml-2 text-[10px] font-extrabold px-2 py-0.5 rounded bg-purple-500/20 text-purple-700 dark:text-purple-300 border border-purple-500/30">
              {badge}
            </span>
          )}
        </div>

        <div className="flex items-center space-x-3 mt-1">
          {Icon && (
            <div className="p-2 rounded-xl bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20 flex-shrink-0">
              <Icon className="w-5 h-5" />
            </div>
          )}
          <div>
            <h1 className="font-heading font-extrabold text-2xl text-slate-900 dark:text-white tracking-tight leading-tight">
              {title}
            </h1>
            {subtitle && (
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 leading-normal">
                {subtitle}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
