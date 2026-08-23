import React, { useState, useEffect, lazy, Suspense } from 'react';
import { WorkspaceProvider } from './context/WorkspaceContext';
import { ThemeProvider } from './context/ThemeContext';
import WorkspaceSelectorModal from './components/WorkspaceSelectorModal';
import CreateWorkspaceModal from './components/CreateWorkspaceModal';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import { fetchHealthStatus } from './api/academicApi';
import { Loader2 } from 'lucide-react';

const LandingPage = lazy(() => import('./pages/LandingPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const UploadWorkspacePage = lazy(() => import('./pages/UploadWorkspacePage'));
const AskPage = lazy(() => import('./pages/AskPage'));
const PYQIntelligencePage = lazy(() => import('./pages/PYQIntelligencePage'));
const StudyPriorityPage = lazy(() => import('./pages/StudyPriorityPage'));
const DemoPage = lazy(() => import('./pages/DemoPage'));

function PageLoader() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[300px] space-y-3 text-purple-400">
      <Loader2 className="w-8 h-8 animate-spin" />
      <span className="text-xs font-semibold tracking-wide text-slate-400">Loading module...</span>
    </div>
  );
}

function AppContent() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [stats, setStats] = useState(null);
  const [selectedTopicForAsk, setSelectedTopicForAsk] = useState('');

  useEffect(() => {
    async function loadStats() {
      const data = await fetchHealthStatus();
      setStats(data);
    }
    loadStats();
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-[#080B14] text-slate-900 dark:text-[#F8FAFC] flex selection:bg-purple-600 selection:text-white transition-colors duration-150">
      
      {/* Workspace Selection & Creation Modals */}
      <WorkspaceSelectorModal />
      <CreateWorkspaceModal />

      {/* 1. Left Sidebar Navigation Shell (240–260px) */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isOpen={mobileSidebarOpen}
        setIsOpen={setMobileSidebarOpen}
      />

      {/* 2. Right Main Layout Canvas */}
      <div className="flex-1 flex flex-col lg:pl-[250px] min-w-0">
        
        {/* Top Header */}
        <Header
          activeTab={activeTab}
          onOpenMobileSidebar={() => setMobileSidebarOpen(true)}
        />

        {/* Main Workspace Body */}
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-8 py-6">
          <Suspense fallback={<PageLoader />}>
            {activeTab === 'landing' && <LandingPage setActiveTab={setActiveTab} />}
            
            {activeTab === 'dashboard' && (
              <DashboardPage
                stats={stats}
                setActiveTab={setActiveTab}
              />
            )}

            {activeTab === 'workspace' && (
              <UploadWorkspacePage setActiveTab={setActiveTab} />
            )}

            {activeTab === 'ask' && (
              <AskPage
                initialQuestion={selectedTopicForAsk}
              />
            )}

            {activeTab === 'pyq-analysis' && (
              <PYQIntelligencePage
                setActiveTab={setActiveTab}
                setSelectedTopicForAsk={setSelectedTopicForAsk}
              />
            )}

            {activeTab === 'study-priority' && (
              <StudyPriorityPage
                setActiveTab={setActiveTab}
                setSelectedTopicForAsk={setSelectedTopicForAsk}
              />
            )}

            {activeTab === 'demo' && (
              <DemoPage
                setActiveTab={setActiveTab}
                setSelectedTopicForAsk={setSelectedTopicForAsk}
              />
            )}
          </Suspense>
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-200 dark:border-[#1F2937] bg-white dark:bg-[#080B14] py-5 text-center text-xs text-slate-500 transition-colors">
          <div className="max-w-7xl mx-auto px-4 sm:px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
            <span>University Academic AI — Grounded Academic Intelligence Platform</span>
            <span>Isolated Academic Workspaces (ChromaDB Scoped)</span>
          </div>
        </footer>

      </div>

    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <WorkspaceProvider>
        <AppContent />
      </WorkspaceProvider>
    </ThemeProvider>
  );
}
