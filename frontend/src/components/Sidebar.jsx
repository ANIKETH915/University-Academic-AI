import React from 'react';
import { useWorkspace } from '../context/WorkspaceContext';
import { 
  LayoutDashboard, 
  Database, 
  Cpu, 
  Flame, 
  GraduationCap, 
  Layers, 
  Sparkles,
  PlayCircle,
  X
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, isOpen, setIsOpen }) {
  const { activeWorkspace, setIsSelectorOpen } = useWorkspace();

  const navItems = [
    { id: 'landing', label: 'Home', icon: Sparkles },
    { id: 'dashboard', label: 'Academic Dashboard', icon: LayoutDashboard },
    { id: 'workspace', label: 'Knowledge Base', icon: Database },
    { id: 'ask', label: 'Ask Academic AI', icon: Cpu, highlight: true },
    { id: 'pyq-analysis', label: 'PYQ Intelligence', icon: Flame },
    { id: 'study-priority', label: 'Study Priority', icon: GraduationCap },
    { id: 'demo', label: 'Preloaded Demo', icon: PlayCircle, badge: 'Sample' }
  ];

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div 
          onClick={() => setIsOpen(false)} 
          className="fixed inset-0 bg-black/70 backdrop-blur-xs z-40 lg:hidden"
        />
      )}

      {/* Sidebar Shell */}
      <aside className={`
        fixed top-0 bottom-0 left-0 z-50 w-[250px] bg-white dark:bg-[#0B1020] border-r border-slate-200 dark:border-[#1F2937]
        flex flex-col justify-between transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        
        {/* Top Header Logo */}
        <div className="p-5 border-b border-slate-200 dark:border-[#1F2937] flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-600 flex items-center justify-center text-white shadow-lg glow-purple">
              <GraduationCap className="w-5 h-5" />
            </div>
            <div>
              <div className="font-heading font-extrabold text-sm tracking-tight text-slate-900 dark:text-white">University AI</div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400 font-medium">Exam Intelligence</div>
            </div>
          </div>

          <button 
            onClick={() => setIsOpen(false)}
            className="lg:hidden text-slate-400 hover:text-white p-1"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Active Workspace Quick Pill */}
        <div className="px-4 py-3 mx-3 my-2 rounded-xl bg-slate-100 dark:bg-[#080B14] border border-slate-200 dark:border-[#1F2937] text-xs flex items-center justify-between">
          <div className="truncate pr-2">
            <div className="text-[9px] uppercase font-bold text-slate-500">Active Workspace</div>
            <div className="font-semibold text-purple-600 dark:text-purple-300 truncate">{activeWorkspace?.subject || 'Select a workspace'}</div>
          </div>
          <button
            onClick={() => setIsSelectorOpen(true)}
            className="p-1 rounded-lg bg-white dark:bg-[#111827] text-purple-600 dark:text-purple-400 hover:text-slate-900 dark:hover:text-white transition-colors border border-slate-200 dark:border-transparent"
            title="Switch Workspace"
          >
            <Layers className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Navigation Items List */}
        <div className="px-3 py-2 flex-1 overflow-y-auto space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;

            return (
              <button
                key={item.id}
                onClick={() => {
                  setActiveTab(item.id);
                  setIsOpen(false);
                }}
                className={`
                  w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold
                  transition-all group relative
                  ${isActive 
                    ? 'bg-purple-600/15 border border-purple-500/40 text-purple-700 dark:text-white shadow-sm' 
                    : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-[#111827] hover:text-slate-900 dark:hover:text-slate-200 border border-transparent'}
                `}
              >
                <div className="flex items-center space-x-2.5">
                  <Icon className={`w-4 h-4 transition-colors ${
                    isActive ? 'text-purple-600 dark:text-purple-400' : 'text-slate-500 group-hover:text-slate-700 dark:group-hover:text-slate-300'
                  }`} />
                  <span>{item.label}</span>
                </div>

                {item.badge && (
                  <span className="px-1.5 py-0.5 rounded-md bg-purple-500/20 text-purple-700 dark:text-purple-300 text-[9px] font-bold uppercase">
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Footer Tagline */}
        <div className="p-4 border-t border-slate-200 dark:border-[#1F2937] text-[11px] text-slate-500 text-center">
          "Your syllabus. Your PYQs."
        </div>

      </aside>
    </>
  );
}
