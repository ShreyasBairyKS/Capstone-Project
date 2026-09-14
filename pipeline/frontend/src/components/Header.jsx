import React from 'react';
import { Play, Settings, RefreshCw, Zap, Cpu, Activity, CheckCircle2, AlertOctagon, UploadCloud } from 'lucide-react';

export default function Header({ 
  isConnected, 
  hardware, 
  actuatorTriggered, 
  onSimulate, 
  isSimulating, 
  autoStream, 
  onToggleAutoStream, 
  onOpenSettings,
  onOpenUpload
}) {
  return (
    <header className="dashboard-header">
      <div className="header-inner">
        <div className="brand-section">
          <div className="brand-logo" style={{
            width: '44px',
            height: '44px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(16, 185, 129, 0.15))',
            border: '1px solid rgba(6, 182, 212, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(6, 182, 212, 0.35)',
            position: 'relative'
          }}>
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="7" stroke="url(#visionq-grad)" />
              <path d="M16 16l4.5 4.5" stroke="#10b981" strokeWidth="2.5" />
              <circle cx="11" cy="11" r="2.5" fill="#06b6d4" />
              <defs>
                <linearGradient id="visionq-grad" x1="4" y1="4" x2="18" y2="18" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#06b6d4" />
                  <stop offset="1" stopColor="#10b981" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div className="brand-text">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <h1 style={{
                margin: 0,
                fontSize: '1.45rem',
                fontWeight: 800,
                letterSpacing: '-0.03em',
                fontFamily: 'var(--font-display)',
                display: 'inline-flex',
                alignItems: 'baseline'
              }}>
                <span style={{ color: '#f8fafc' }}>Vision</span>
                <span style={{ color: '#06b6d4', textShadow: '0 0 12px rgba(6, 182, 212, 0.6)' }}>Q</span>
                <span style={{ color: '#10b981', textShadow: '0 0 12px rgba(16, 185, 129, 0.5)' }}>AI</span>
              </h1>
              <span style={{
                fontSize: '0.65rem',
                fontWeight: 700,
                letterSpacing: '0.08em',
                padding: '2px 6px',
                borderRadius: '4px',
                background: 'rgba(6, 182, 212, 0.15)',
                border: '1px solid rgba(6, 182, 212, 0.35)',
                color: '#38bdf8',
                textTransform: 'uppercase'
              }}>PRO</span>
            </div>
            <p style={{ margin: '2px 0 0', fontSize: '0.78rem', color: '#94a3b8', letterSpacing: '0.01em' }}>
              Autonomous Defect Inspection System &bull; YOLOv11m-seg Distilled
            </p>
          </div>
        </div>

        <div className="header-status-group">
          {/* WebSocket Status */}
          <div className={`chip ${isConnected ? 'online' : 'offline'}`}>
            <span className="pulsing-dot" />
            <span>{isConnected ? 'Stream Connected' : 'Connecting...'}</span>
          </div>

          {/* Hardware Engine */}
          <div className="chip hardware">
            <Cpu size={14} />
            <span>{hardware || 'CPU Fallback (14T)'}</span>
          </div>

          {/* Actuator Status */}
          <div className={`chip ${actuatorTriggered ? 'actuator-triggered' : 'actuator-standby'}`}>
            {actuatorTriggered ? <AlertOctagon size={14} /> : <CheckCircle2 size={14} />}
            <span>{actuatorTriggered ? 'ACTUATOR: EJECT TRIGGERED' : 'ACTUATOR: STANDBY'}</span>
          </div>

          {/* Upload Images from Computer */}
          <button 
            className="btn btn-secondary" 
            onClick={onOpenUpload}
            style={{ borderColor: 'rgba(6, 182, 212, 0.4)', background: 'rgba(6, 182, 212, 0.08)' }}
            title="Upload custom image files from your computer"
          >
            <UploadCloud size={16} className="text-cyan-400" />
            <span>Upload & Inspect</span>
          </button>

          {/* Simulate Button */}
          <button 
            className="btn btn-primary" 
            onClick={onSimulate} 
            disabled={isSimulating}
          >
            <Play size={16} />
            <span>{isSimulating ? 'Inspecting...' : 'Simulate Bottle'}</span>
          </button>

          {/* Auto Conveyor Runner */}
          <button 
            className={`btn ${autoStream ? 'btn-danger' : 'btn-secondary'}`}
            onClick={onToggleAutoStream}
          >
            <Activity size={16} />
            <span>{autoStream ? 'Stop Conveyor' : 'Run Conveyor'}</span>
          </button>

          {/* Settings Modal Toggle */}
          <button className="btn btn-secondary" onClick={onOpenSettings} title="Pipeline Settings">
            <Settings size={16} />
          </button>
        </div>
      </div>
    </header>
  );
}
