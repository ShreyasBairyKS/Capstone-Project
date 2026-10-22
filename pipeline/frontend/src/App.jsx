import React, { useState, useEffect, useRef } from 'react';
import { ThemeProvider } from './context/ThemeContext';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import KpiBar from './components/KpiBar';
import MultiCameraFeed from './components/MultiCameraFeed';
import ConveyorFilmstrip from './components/ConveyorFilmstrip';
import DefectAnalyticsWidget from './components/DefectAnalyticsWidget';
import DefectAuditTable from './components/DefectAuditTable';
import DefectAuditPage from './components/DefectAuditPage';
import ReplayInspectorPage from './components/ReplayInspectorPage';
import ActiveLearningPage from './components/ActiveLearningPage';
import ReplayModal from './components/ReplayModal';
import SettingsModal from './components/SettingsModal';
import UploadModal from './components/UploadModal';

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [hardware, setHardware] = useState('CPU Fallback (14T)');
  const [stats, setStats] = useState(null);
  const [latestInspection, setLatestInspection] = useState(null);
  const [inspectionHistory, setInspectionHistory] = useState([]);
  const [defects, setDefects] = useState([]);
  const [actuatorTriggered, setActuatorTriggered] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [autoStream, setAutoStream] = useState(false);
  const urlParams = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : new URLSearchParams();
  const initialTab = urlParams.get('tab') || 'dashboard';
  const initialBottle = urlParams.get('bottle_id') || null;

  const [selectedBottleId, setSelectedBottleId] = useState(initialBottle);
  const [modalBottleId, setModalBottleId] = useState(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(urlParams.get('modal') === 'settings');
  const [isUploadOpen, setIsUploadOpen] = useState(urlParams.get('modal') === 'upload');
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState(initialTab);

  const handleSelectTab = (tab) => {
    setActiveTab(tab);
    try {
      const url = new URL(window.location);
      url.searchParams.set('tab', tab);
      window.history.replaceState({}, '', url);
    } catch (_) {}
  };

  const handleSelectBottle = (id) => {
    setSelectedBottleId(id);
    try {
      const url = new URL(window.location);
      if (id) {
        url.searchParams.set('bottle_id', id);
      } else {
        url.searchParams.delete('bottle_id');
      }
      window.history.replaceState({}, '', url);
    } catch (_) {}
  };

  const wsRef = useRef(null);
  const autoStreamIntervalRef = useRef(null);

  // 1. Initialize WebSocket Connection
  useEffect(() => {
    let reconnectTimer = null;

    const connectWs = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        console.log('Connected to Inspection Stream WebSocket');
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'connection_established') {
            if (msg.hardware) setHardware(msg.hardware);
            if (msg.stats) setStats(msg.stats);
          } else if (msg.type === 'inspection_cycle') {
            handleInspectionReceived(msg.data, msg.stats);
          } else if (msg.type === 'stats') {
            setStats(msg.stats);
          }
        } catch (err) {
          console.error('Error parsing WS message:', err);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        reconnectTimer = setTimeout(connectWs, 3000);
      };

      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        ws.close();
      };
    };

    connectWs();
    fetchInitialData();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };
  }, []);

  const handleInspectionReceived = (inspData, updatedStats) => {
    setLatestInspection(inspData);
    if (updatedStats) setStats(updatedStats);

    // Append to inspection history queue (keep last 20 bottles)
    setInspectionHistory(prev => {
      const updated = [inspData, ...prev.filter(b => b.bottle_id !== inspData.bottle_id)];
      return updated.slice(0, 20);
    });

    // Actuator flash indicator
    if (inspData.trigger_reject_actuator) {
      setActuatorTriggered(true);
      setTimeout(() => setActuatorTriggered(false), 2200);
    }

    // Refresh defect table
    fetchDefects();
  };

  const fetchInitialData = async () => {
    try {
      const [hRes, sRes, dRes] = await Promise.all([
        fetch('/api/health').then(r => r.json()).catch(() => null),
        fetch('/api/stats').then(r => r.json()).catch(() => null),
        fetch('/api/defects?limit=25').then(r => r.json()).catch(() => null)
      ]);
      if (hRes?.hardware) setHardware(hRes.hardware);
      if (sRes) setStats(sRes);
      if (Array.isArray(dRes)) {
        setDefects(dRes);
      } else if (dRes && Array.isArray(dRes.items)) {
        setDefects(dRes.items);
      }
    } catch (err) {
      console.error('Error fetching initial data:', err);
    }
  };

  const fetchDefects = async () => {
    try {
      const res = await fetch('/api/defects?limit=25');
      if (res.ok) {
        const data = await res.json();
        setDefects(data.items || data);
      }
    } catch (err) {
      console.error('Error fetching defects:', err);
    }
  };

  // 2. Trigger Inspection Simulation
  const handleSimulate = async () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      setIsSimulating(true);
      wsRef.current.send(JSON.stringify({ command: 'simulate' }));
      setTimeout(() => setIsSimulating(false), 600);
    } else {
      setIsSimulating(true);
      try {
        const res = await fetch('/api/simulate', { method: 'POST' });
        if (res.ok) {
          const data = await res.json();
          handleInspectionReceived(data);
          fetchInitialData();
        }
      } catch (err) {
        console.error('Error running simulate via REST:', err);
      } finally {
        setIsSimulating(false);
      }
    }
  };

  // 3. Handle Manual User Upload Inspection
  const handleManualUploadComplete = (inspectResult) => {
    handleInspectionReceived(inspectResult);
    fetchInitialData();
  };

  // 4. Auto-Conveyor Stream Ticker
  const handleToggleAutoStream = () => {
    if (autoStream) {
      clearInterval(autoStreamIntervalRef.current);
      setAutoStream(false);
    } else {
      setAutoStream(true);
      handleSimulate();
      autoStreamIntervalRef.current = setInterval(() => {
        handleSimulate();
      }, 3500);
    }
  };

  useEffect(() => {
    return () => {
      if (autoStreamIntervalRef.current) clearInterval(autoStreamIntervalRef.current);
    };
  }, []);

  return (
    <ThemeProvider>
      <div className="flex min-h-screen bg-[#050811] text-[#dae2fd] dark:bg-[#050811] dark:text-[#dae2fd] light:bg-[#f1f5f9] light:text-[#0f172a] transition-colors duration-200">
        {/* Collapsible Industrial Sidebar */}
        <Sidebar
          activeTab={activeTab}
          onSelectTab={handleSelectTab}
          isCollapsed={isSidebarCollapsed}
          onToggleCollapse={() => setIsSidebarCollapsed(prev => !prev)}
          onOpenUpload={() => setIsUploadOpen(true)}
          onOpenSettings={() => setIsSettingsOpen(true)}
          hardware={hardware}
          defectCount={defects.length}
        />

        {/* Main Canvas Workspace */}
        <div 
          className={`flex-1 flex flex-col min-w-0 transition-all duration-200
            ${isSidebarCollapsed ? 'ml-16' : 'ml-64'}`
          }
        >
          {/* M1: Global System Status & Header Bar */}
          <Header 
            isConnected={isConnected}
            hardware={hardware}
            actuatorTriggered={actuatorTriggered}
            onSimulate={handleSimulate}
            isSimulating={isSimulating}
            autoStream={autoStream}
            onToggleAutoStream={handleToggleAutoStream}
            onOpenSettings={() => setIsSettingsOpen(true)}
            onOpenUpload={() => setIsUploadOpen(true)}
          />

          <main className="p-3.5 flex flex-col gap-3.5 max-w-[1920px] mx-auto w-full">
            {/* View Switching: Default is Main Operational Dashboard */}
            {activeTab === 'dashboard' && (
              <>
                {/* M2: Production & Quality KPI Tiles */}
                <KpiBar 
                  stats={stats} 
                  latestLatency={latestInspection?.total_latency_ms} 
                />

                {/* M3: Live Synchronized Dual-Camera Viewport */}
                <MultiCameraFeed 
                  latestInspection={latestInspection} 
                  onInspectBottle={id => {
                    handleSelectBottle(id);
                    handleSelectTab('replay');
                  }}
                />

                {/* M4: Real-Time Conveyor Filmstrip Queue */}
                <ConveyorFilmstrip 
                  inspectionHistory={inspectionHistory}
                  onSelectBottle={id => {
                    handleSelectBottle(id);
                    handleSelectTab('replay');
                  }}
                />

                {/* M5 & M6: Recharts Defect Pareto & Hourly Trend */}
                <DefectAnalyticsWidget 
                  stats={stats}
                  defects={defects}
                />
              </>
            )}

            {/* Standalone Defect Audit Log Tab (S1) */}
            {activeTab === 'audit' && (
              <DefectAuditPage 
                onSelectBottle={id => {
                  handleSelectBottle(id);
                  handleSelectTab('replay');
                }} 
              />
            )}

            {/* Deep-Dive Bottle Replay & Contour Inspector Workstation (S2) */}
            {activeTab === 'replay' && (
              <ReplayInspectorPage 
                initialBottleId={selectedBottleId}
                onSelectBottle={id => handleSelectBottle(id)}
                onTriageToAnnotation={id => {
                  handleSelectBottle(id);
                  handleSelectTab('annotation');
                }}
              />
            )}

            {/* Active Learning & Annotation Workbench (S3) */}
            {activeTab === 'annotation' && (
              <ActiveLearningPage 
                initialBottleId={selectedBottleId}
                onSelectBottle={id => handleSelectBottle(id)} 
              />
            )}
          </main>
        </div>

        {/* Modals */}
        <UploadModal
          isOpen={isUploadOpen}
          onClose={() => setIsUploadOpen(false)}
          onInspectionComplete={handleManualUploadComplete}
        />

        <ReplayModal 
          bottleId={modalBottleId} 
          onClose={() => setModalBottleId(null)} 
        />

        <SettingsModal 
          isOpen={isSettingsOpen} 
          onClose={() => setIsSettingsOpen(false)} 
        />
      </div>
    </ThemeProvider>
  );
}
