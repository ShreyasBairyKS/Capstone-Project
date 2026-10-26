import React, { useState, useEffect, useRef, useCallback } from 'react';
import { ThemeProvider } from './context/ThemeContext';
import { StationProvider, useStation } from './context/StationContext';
import Sidebar from './components/Sidebar';
import Header from './components/Header';

// Station 1: Cap Inspection Components
import KpiBar from './components/KpiBar';
import MultiCameraFeed from './components/MultiCameraFeed';
import ConveyorFilmstrip from './components/ConveyorFilmstrip';
import DefectAnalyticsWidget from './components/DefectAnalyticsWidget';
import DefectAuditPage from './components/DefectAuditPage';
import ReplayInspectorPage from './components/ReplayInspectorPage';
import ActiveLearningPage from './components/ActiveLearningPage';

// Station 2: Fill Level Inspection Components
import FillDashboard from './components/fill_level/FillDashboard';
import FillAuditPage from './components/fill_level/FillAuditPage';
import FillReplayInspectorPage from './components/fill_level/FillReplayInspectorPage';
import FillActiveLearningPage from './components/fill_level/FillActiveLearningPage';
import FillUploadModal from './components/fill_level/FillUploadModal';
import FillSettingsModal from './components/fill_level/FillSettingsModal';

// Modals
import ReplayModal from './components/ReplayModal';
import SettingsModal from './components/SettingsModal';
import UploadModal from './components/UploadModal';
import { Loader2 } from 'lucide-react';

function MainAppContent() {
  const { activeStation, isSwapping, swapMessage, swapError } = useStation();

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

  // Fetch initial telemetry for active station
  const fetchInitialData = useCallback(async () => {
    try {
      const isFill = activeStation === 'fill';
      const healthEndpoint = isFill ? '/api/fill/health' : '/api/health';
      const statsEndpoint = isFill ? '/api/fill/stats' : '/api/stats';
      const defectsEndpoint = isFill ? '/api/fill/defects?limit=25&verdict=REJECT' : '/api/defects?limit=25';

      const [hRes, sRes, dRes] = await Promise.all([
        fetch(healthEndpoint).then(r => r.json()).catch(() => null),
        fetch(statsEndpoint).then(r => r.json()).catch(() => null),
        fetch(defectsEndpoint).then(r => r.json()).catch(() => null)
      ]);

      if (hRes) {
        const hw = hRes.hardware || hRes.hardware_runtime;
        if (hw) setHardware(hw);
      }
      if (sRes) setStats(sRes);
      if (Array.isArray(dRes)) {
        setDefects(dRes);
      } else if (dRes && Array.isArray(dRes.items)) {
        setDefects(dRes.items);
      }
    } catch (err) {
      console.error('Error fetching initial station data:', err);
    }
  }, [activeStation]);

  const fetchDefects = useCallback(async () => {
    try {
      const isFill = activeStation === 'fill';
      const endpoint = isFill ? '/api/fill/defects?limit=25&verdict=REJECT' : '/api/defects?limit=25';
      const res = await fetch(endpoint);
      if (res.ok) {
        const data = await res.json();
        setDefects(data.items || data);
      }
    } catch (err) {
      console.error('Error fetching defects:', err);
    }
  }, [activeStation]);

  const handleInspectionReceived = useCallback((inspData, updatedStats) => {
    setLatestInspection(inspData);
    if (updatedStats) setStats(updatedStats);

    // Append to inspection history queue (keep last 20 bottles)
    setInspectionHistory(prev => {
      const updated = [inspData, ...prev.filter(b => b.bottle_id !== inspData.bottle_id)];
      return updated.slice(0, 20);
    });

    // Actuator flash indicator
    if (inspData.trigger_reject_actuator || inspData.trigger_actuator) {
      setActuatorTriggered(true);
      setTimeout(() => setActuatorTriggered(false), 2200);
    }

    fetchDefects();
  }, [fetchDefects]);

  // Connect WebSocket dynamically based on activeStation
  useEffect(() => {
    let reconnectTimer = null;
    let isSubscribed = true;

    // Stop the auto-conveyor ticker: its handler is bound to the previous station's endpoints
    if (autoStreamIntervalRef.current) {
      clearInterval(autoStreamIntervalRef.current);
      autoStreamIntervalRef.current = null;
    }
    setAutoStream(false);

    // Reset station views on switch
    setLatestInspection(null);
    setInspectionHistory([]);
    setSelectedBottleId(null);
    setStats(null);
    setDefects([]);

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    const connectWs = () => {
      if (!isSubscribed) return;
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const path = activeStation === 'fill' ? '/ws/fill/stream' : '/ws/stream';
      const wsUrl = `${protocol}//${window.location.host}${path}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!isSubscribed) return;
        setIsConnected(true);
        console.log(`Connected to Station WebSocket: ${path}`);
      };

      ws.onmessage = (event) => {
        if (!isSubscribed) return;
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
        if (!isSubscribed) return;
        setIsConnected(false);
        reconnectTimer = setTimeout(connectWs, 3000);
      };

      ws.onerror = (err) => {
        console.warn('WebSocket connection error:', err);
        ws.close();
      };
    };

    connectWs();
    fetchInitialData();

    return () => {
      isSubscribed = false;
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };
  }, [activeStation, fetchInitialData, handleInspectionReceived]);

  // Trigger Inspection Simulation
  const handleSimulate = async () => {
    const isFill = activeStation === 'fill';
    setIsSimulating(true);

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'simulate' }));
      setTimeout(() => setIsSimulating(false), 600);
    } else {
      try {
        const endpoint = isFill ? '/api/fill/simulate' : '/api/simulate';
        const res = await fetch(endpoint, { method: 'POST' });
        if (res.ok) {
          const json = await res.json();
          const inspData = json.data || json;
          const statData = json.stats;
          handleInspectionReceived(inspData, statData);
          fetchInitialData();
        }
      } catch (err) {
        console.error('Error running simulate cycle:', err);
      } finally {
        setIsSimulating(false);
      }
    }
  };

  // Manual User Upload Inspection Handler
  const handleManualUploadComplete = (inspectResult) => {
    handleInspectionReceived(inspectResult);
    fetchInitialData();
  };

  // Auto-Conveyor Stream Ticker
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
        {/* Global System Status & Header Bar */}
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

        {/* Dynamic VRAM Swap Transition Banner */}
        {isSwapping && (
          <div className="bg-[#17253d] border-b border-[#00daf3]/50 px-4 py-2 flex items-center justify-center gap-2 text-xs font-mono text-[#00daf3] shadow-md animate-pulse">
            <Loader2 size={14} className="animate-spin" />
            <span className="font-bold">DYNAMIC GPU VRAM SWAP:</span>
            <span>{swapMessage || 'Switching active vision models on GPU...'}</span>
          </div>
        )}

        {swapError && !isSwapping && (
          <div className="bg-[#3a1512] border-b border-[#ff3b30]/60 px-4 py-2 text-xs font-mono text-[#ff817b] text-center">
            {swapError}
          </div>
        )}

        <main className="p-3.5 flex flex-col gap-3.5 max-w-[1920px] mx-auto w-full">
          {/* ================================================================= */}
          {/* STATION 1: BOTTLE CAP DEFECT INSPECTION                           */}
          {/* ================================================================= */}
          {activeStation === 'cap' && (
            <>
              {activeTab === 'dashboard' && (
                <>
                  <KpiBar 
                    stats={stats} 
                    latestLatency={latestInspection?.total_latency_ms} 
                  />
                  <MultiCameraFeed 
                    latestInspection={latestInspection} 
                    onInspectBottle={id => {
                      handleSelectBottle(id);
                      handleSelectTab('replay');
                    }}
                  />
                  <ConveyorFilmstrip 
                    inspectionHistory={inspectionHistory}
                    onSelectBottle={id => {
                      handleSelectBottle(id);
                      handleSelectTab('replay');
                    }}
                  />
                  <DefectAnalyticsWidget 
                    stats={stats}
                    defects={defects}
                  />
                </>
              )}

              {activeTab === 'audit' && (
                <DefectAuditPage 
                  onSelectBottle={id => {
                    handleSelectBottle(id);
                    handleSelectTab('replay');
                  }} 
                />
              )}

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

              {activeTab === 'annotation' && (
                <ActiveLearningPage 
                  initialBottleId={selectedBottleId}
                  onSelectBottle={id => handleSelectBottle(id)} 
                />
              )}
            </>
          )}

          {/* ================================================================= */}
          {/* STATION 2: BOTTLE FILL LEVEL INSPECTION                           */}
          {/* ================================================================= */}
          {activeStation === 'fill' && (
            <>
              {activeTab === 'dashboard' && (
                <FillDashboard
                  stats={stats}
                  latestInspection={latestInspection}
                  inspectionHistory={inspectionHistory}
                  defects={defects}
                  onSelectBottle={id => {
                    handleSelectBottle(id);
                    handleSelectTab('replay');
                  }}
                />
              )}

              {activeTab === 'audit' && (
                <FillAuditPage 
                  onSelectBottle={id => {
                    handleSelectBottle(id);
                    handleSelectTab('replay');
                  }}
                />
              )}

              {activeTab === 'replay' && (
                <FillReplayInspectorPage 
                  initialBottleId={selectedBottleId}
                  onSelectBottle={handleSelectBottle}
                  onTriageToAnnotation={id => {
                    handleSelectBottle(id);
                    handleSelectTab('annotation');
                  }}
                />
              )}

              {activeTab === 'annotation' && (
                <FillActiveLearningPage 
                  initialBottleId={selectedBottleId}
                  onSelectBottle={handleSelectBottle}
                />
              )}
            </>
          )}
        </main>
      </div>

      {/* Global Modals */}
      {activeStation === 'fill' ? (
        <FillUploadModal
          isOpen={isUploadOpen}
          onClose={() => setIsUploadOpen(false)}
          onInspectionComplete={handleManualUploadComplete}
        />
      ) : (
        <UploadModal
          isOpen={isUploadOpen}
          onClose={() => setIsUploadOpen(false)}
          onInspectionComplete={handleManualUploadComplete}
        />
      )}

      <ReplayModal 
        bottleId={modalBottleId} 
        onClose={() => setModalBottleId(null)} 
      />

      {/* Quality-gate settings are station-specific: each station has its own modal and /config API */}
      {activeStation === 'fill' ? (
        <FillSettingsModal
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
        />
      ) : (
        <SettingsModal 
          isOpen={isSettingsOpen} 
          onClose={() => setIsSettingsOpen(false)} 
        />
      )}
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <StationProvider>
        <MainAppContent />
      </StationProvider>
    </ThemeProvider>
  );
}
