import React, { useState, useEffect } from 'react';
import { WorkspaceProvider } from './context/WorkspaceContext';
import WorkspaceSelectorModal from './components/WorkspaceSelectorModal';
import CreateWorkspaceModal from './components/CreateWorkspaceModal';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import LandingPage from './pages/LandingPage';
import DashboardPage from './pages/DashboardPage';
import UploadWorkspacePage from './pages/UploadWorkspacePage';
import AskPage from './pages/AskPage';
import PYQIntelligencePage from './pages/PYQIntelligencePage';
import StudyPriorityPage from './pages/StudyPriorityPage';
import DemoPage from './pages/DemoPage';
import { fetchHealthStatus } from './api/academicApi';

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
    <div className="min-h-screen bg-[#080B14] text-[#F8FAFC] flex selection:bg-purple-600 selection:text-white">
      
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
        </main>

        {/* Footer */}
        <footer className="border-t border-[#1F2937] bg-[#080B14] py-5 text-center text-xs text-slate-500">
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
    <WorkspaceProvider>
      <AppContent />
    </WorkspaceProvider>
  );
}
