import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import KpiBar from './components/KpiBar';
import MultiCameraFeed from './components/MultiCameraFeed';
import DefectAuditTable from './components/DefectAuditTable';
import ReplayModal from './components/ReplayModal';
import SettingsModal from './components/SettingsModal';
import UploadModal from './components/UploadModal';

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [hardware, setHardware] = useState('CPU Fallback (14T)');
  const [stats, setStats] = useState(null);
  const [latestInspection, setLatestInspection] = useState(null);
  const [defects, setDefects] = useState([]);
  const [actuatorTriggered, setActuatorTriggered] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [autoStream, setAutoStream] = useState(false);
  const [selectedBottleId, setSelectedBottleId] = useState(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);

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
            const inspData = msg.data;
            handleInspectionReceived(inspData, msg.stats);
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
        console.error('WebSocket encountered error:', err);
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

    // Actuator flash indicator
    if (inspData.trigger_reject_actuator) {
      setActuatorTriggered(true);
      setTimeout(() => setActuatorTriggered(false), 2500);
    }

    // Refresh defect list
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
      if (Array.isArray(dRes)) setDefects(dRes);
    } catch (err) {
      console.error('Error fetching initial data:', err);
    }
  };

  const fetchDefects = async () => {
    try {
      const res = await fetch('/api/defects?limit=25');
      if (res.ok) {
        const data = await res.json();
        setDefects(data);
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
      setTimeout(() => setIsSimulating(false), 800);
    } else {
      // Fallback to REST endpoint
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
    <div className="app-container">
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

      <main className="main-content">
        {/* Real-time KPI Bar */}
        <KpiBar 
          stats={stats} 
          latestLatency={latestInspection?.total_latency_ms} 
        />

        {/* Live Multi-Camera Inspection Viewport */}
        <MultiCameraFeed 
          latestInspection={latestInspection} 
        />

        {/* Defect Audit & Active Learning Log */}
        <DefectAuditTable 
          defects={defects} 
          onSelectBottle={id => setSelectedBottleId(id)} 
        />
      </main>

      {/* Manual File Upload Inspection Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onInspectionComplete={handleManualUploadComplete}
      />

      {/* Dynamic Replay Modal */}
      <ReplayModal 
        bottleId={selectedBottleId} 
        onClose={() => setSelectedBottleId(null)} 
      />

      {/* Threshold Settings Modal */}
      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)} 
      />
    </div>
  );
}
