import React from 'react';
import { 
  Play, 
  Pause, 
  RefreshCw, 
  Zap, 
  Cpu, 
  Activity, 
  CheckCircle2, 
  AlertOctagon, 
  UploadCloud, 
  Settings, 
  Sun, 
  Moon,
  Radio
} from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

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
  const { theme, toggleTheme, isDark } = useTheme();

  return (
    <header className="sticky top-0 z-30 w-full h-14 bg-[#060e20] text-[#dae2fd] border-b border-[#1e2e48] px-4 flex items-center justify-between gap-3 shadow-md">
      {/* Left: Real-Time Line State & Telemetry Strip */}
      <div className="flex-1 min-w-0 flex items-center gap-2.5 overflow-hidden py-1">
        {/* Stream Link State */}
        <div className={`flex items-center gap-2 px-2.5 py-1 rounded text-xs font-mono font-semibold border shrink-0
          ${isConnected 
            ? 'bg-[#00f5a0]/10 text-[#00f5a0] border-[#00f5a0]/30' 
            : 'bg-[#ff3b30]/10 text-[#ff3b30] border-[#ff3b30]/30'}`
        }>
          <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-[#00f5a0] animate-pulse' : 'bg-[#ff3b30]'}`} />
          <span>{isConnected ? 'FEED: LIVE' : 'FEED: CONNECTING'}</span>
        </div>

        {/* Operating Line State */}
        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#171f33] border border-[#1e2e48] text-xs font-mono shrink-0">
          <span className="text-[#849588]">LINE:</span>
          <span className="text-[#00daf3] font-bold">
            <span className="xl:inline hidden">04 HIGH-SPEED CAP</span>
            <span className="xl:hidden inline">04</span>
          </span>
        </div>

        {/* Hardware Acceleration Badge (Shown on 2xl+ to save space on laptops) */}
        <div className="hidden 2xl:flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#131b2e] border border-[#1e2e48] text-xs font-mono text-[#94a3b8] shrink-0">
          <Cpu size={14} className="text-[#00daf3]" />
          <span>{hardware || 'CPU Fallback (14T)'}</span>
        </div>

        {/* Physical Actuator Status */}
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono font-bold border transition-all duration-150 shrink-0
          ${actuatorTriggered 
            ? 'bg-[#ff3b30] text-white border-[#ff3b30] shadow-[0_0_14px_rgba(255,59,48,0.7)] animate-bounce' 
            : 'bg-[#171f33] text-[#94a3b8] border-[#1e2e48]'}`
        }>
          {actuatorTriggered ? <AlertOctagon size={14} /> : <CheckCircle2 size={14} className="text-[#00f5a0]" />}
          <span>
            {actuatorTriggered 
              ? 'ACTUATOR: EJECT FIRED' 
              : (
                <>
                  <span className="xl:inline hidden">ACTUATOR: ARMED (12 PSI)</span>
                  <span className="xl:hidden inline">ACTUATOR: ARMED</span>
                </>
              )
            }
          </span>
        </div>
      </div>

      {/* Right: Operational Controls & Theme Switcher */}
      <div className="flex items-center gap-2 flex-shrink-0">
        {/* Run / Stop Conveyor Toggle */}
        <button
          onClick={onToggleAutoStream}
          className={`h-8 px-3 rounded flex items-center gap-1.5 text-xs font-mono font-semibold transition-all duration-150 border
            ${autoStream 
              ? 'bg-[#ff3b30]/20 text-[#ff817b] border-[#ff3b30] hover:bg-[#ff3b30]/30 shadow-[0_0_8px_rgba(255,59,48,0.3)]' 
              : 'bg-[#171f33] text-[#00f5a0] border-[#00f5a0]/40 hover:bg-[#1f2c47] hover:border-[#00f5a0]'}`
          }
          title="Toggle continuous conveyor bottle inspection"
        >
          {autoStream ? <Pause size={14} /> : <Activity size={14} />}
          <span className="hidden sm:inline">{autoStream ? 'PAUSE CONVEYOR' : 'RUN CONVEYOR'}</span>
        </button>

        {/* Simulate Single Bottle Trigger */}
        <button
          onClick={onSimulate}
          disabled={isSimulating}
          className="h-8 px-3 rounded bg-gradient-to-r from-[#0284c7] to-[#00daf3] hover:from-[#0369a1] hover:to-[#00b4cc] text-[#050811] font-mono font-bold text-xs flex items-center gap-1.5 transition-all shadow-[0_0_10px_rgba(0,218,243,0.3)] disabled:opacity-50"
          title="Trigger a single synchronized inspection cycle"
        >
          <Play size={14} className={isSimulating ? 'animate-spin' : ''} />
          <span>{isSimulating ? 'TRIGGERING...' : 'SIMULATE'}</span>
        </button>

        {/* Upload Custom Image Inspection */}
        <button
          onClick={onOpenUpload}
          className="h-8 px-2.5 rounded bg-[#171f33] hover:bg-[#1f2c47] border border-[#1e2e48] text-[#00daf3] text-xs font-mono flex items-center gap-1.5 transition-colors"
          title="Upload image files for offline quality evaluation"
        >
          <UploadCloud size={14} />
          <span className="hidden md:inline">UPLOAD</span>
        </button>

        {/* Theme Toggle Button (Dark / Light) */}
        <button
          onClick={toggleTheme}
          className="w-8 h-8 rounded bg-[#171f33] hover:bg-[#1f2c47] border border-[#1e2e48] flex items-center justify-center text-[#dae2fd] transition-colors"
          title={`Switch to ${isDark ? 'Light' : 'Dark'} Mode`}
        >
          {isDark ? <Sun size={15} className="text-[#fbbf24]" /> : <Moon size={15} className="text-[#38bdf8]" />}
        </button>

        {/* Settings Modal Toggle */}
        <button
          onClick={onOpenSettings}
          className="w-8 h-8 rounded bg-[#171f33] hover:bg-[#1f2c47] border border-[#1e2e48] flex items-center justify-center text-[#94a3b8] hover:text-[#00f5a0] transition-colors"
          title="Pipeline Threshold Settings"
        >
          <Settings size={15} />
        </button>
      </div>
    </header>
  );
}
