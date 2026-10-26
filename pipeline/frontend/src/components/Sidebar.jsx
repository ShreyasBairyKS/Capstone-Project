import React from 'react';
import { 
  Layers, 
  Activity, 
  Eye, 
  BrainCircuit, 
  UploadCloud, 
  Settings, 
  ChevronLeft, 
  ChevronRight,
  ShieldCheck,
  Cpu
} from 'lucide-react';
import StationSwitcher from './common/StationSwitcher';
import { useStation } from '../context/StationContext';

export default function Sidebar({ 
  activeTab, 
  onSelectTab, 
  isCollapsed, 
  onToggleCollapse,
  onOpenUpload,
  onOpenSettings,
  hardware,
  defectCount = 0
}) {
  const { activeStation } = useStation();

  const navItems = [
    { 
      id: 'dashboard', 
      label: activeStation === 'fill' ? 'Fill Dashboard' : 'Operational Dashboard', 
      icon: Activity 
    },
    { 
      id: 'audit', 
      label: activeStation === 'fill' ? 'Fill Audit Log' : 'Defect Audit Log', 
      icon: Layers 
    },
    { 
      id: 'replay', 
      label: activeStation === 'fill' ? 'Fill Replay Inspector' : 'Replay Inspector', 
      icon: Eye 
    },
    { 
      id: 'annotation', 
      label: activeStation === 'fill' ? 'Fill Active Learning' : 'Active Learning', 
      icon: BrainCircuit 
    },
    { id: 'lab', label: 'Lab & Diagnostics', icon: UploadCloud, action: onOpenUpload },
    { id: 'settings', label: 'Quality Settings', icon: Settings, action: onOpenSettings },
  ];

  return (
    <aside 
      className={`fixed left-0 top-0 h-screen transition-all duration-200 z-40 flex flex-col justify-between py-3 px-2 border-r select-none
        ${isCollapsed ? 'w-16' : 'w-64'} 
        bg-[#0b1326] text-[#dae2fd] border-[#1e2e48] 
        dark:bg-[#060e20] dark:border-[#1e2e48] 
        light:bg-[#f8fafc] light:text-[#0f172a] light:border-[#e2e8f0]`
      }
    >
      {/* Top Brand Header */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between px-2 h-10 overflow-hidden">
          <div className="flex items-center gap-2.5">
            <div className="relative w-9 h-9 rounded bg-[#171f33] border border-[#00f5a0]/40 flex items-center justify-center flex-shrink-0 shadow-[0_0_10px_rgba(0,245,160,0.25)]">
              <ShieldCheck className="text-[#00f5a0]" size={20} />
              <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#00f5a0] animate-ping" />
            </div>
            {!isCollapsed && (
              <div className="whitespace-nowrap overflow-hidden">
                <div className="font-['Outfit'] font-bold text-base tracking-tight text-[#00f5a0] flex items-center gap-1.5">
                  <span>VisionQ</span>
                  <span className="text-[#00daf3] text-xs px-1 py-0.5 rounded bg-[#00daf3]/10 border border-[#00daf3]/30">AI PRO</span>
                </div>
                <div className="text-[10px] font-mono text-[#849588] tracking-wider">EDGE QC MATRIX v4.8</div>
              </div>
            )}
          </div>
          {!isCollapsed && (
            <button 
              onClick={onToggleCollapse}
              className="text-[#849588] hover:text-[#00f5a0] p-1 rounded hover:bg-[#171f33] transition-colors"
              title="Collapse Sidebar"
            >
              <ChevronLeft size={16} />
            </button>
          )}
        </div>

        {/* Collapsed expand toggle */}
        {isCollapsed && (
          <div className="flex justify-center">
            <button 
              onClick={onToggleCollapse}
              className="text-[#849588] hover:text-[#00f5a0] p-1 rounded hover:bg-[#171f33] transition-colors"
              title="Expand Sidebar"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        )}

        {/* Station Switcher: Station 1 (Cap) vs Station 2 (Fill) */}
        <div className="px-1 mt-1">
          <StationSwitcher isCollapsed={isCollapsed} />
        </div>

        {/* Navigation Items */}
        <nav className="flex flex-col gap-1 mt-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  if (item.action) {
                    item.action();
                  } else {
                    onSelectTab(item.id);
                  }
                }}
                title={item.label}
                className={`flex items-center gap-3 px-3 py-2.5 rounded transition-colors text-left font-['Inter'] text-xs font-medium tracking-wide
                  ${isActive 
                    ? 'bg-[#171f33] text-[#00f5a0] border-l-2 border-[#00f5a0] shadow-[0_0_8px_rgba(0,245,160,0.15)] font-semibold' 
                    : 'text-[#94a3b8] hover:text-[#dae2fd] hover:bg-[#131b2e]'
                  }`
                }
              >
                <Icon size={18} className={`flex-shrink-0 ${isActive ? 'text-[#00f5a0]' : 'text-[#849588]'}`} />
                {!isCollapsed && (
                  <span className="whitespace-nowrap overflow-hidden flex-1">{item.label}</span>
                )}
                {!isCollapsed && item.id === 'audit' && defectCount > 0 && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-[#ff3b30]/20 text-[#ff817b] border border-[#ff3b30]/40">
                    {defectCount}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Section: Telemetry & Operator Info */}
      <div className="flex flex-col gap-2 pt-2 border-t border-[#1e2e48]">
        {/* Hardware Status Tile */}
        <div className="p-2 rounded bg-[#060e20] border border-[#1e2e48] flex items-center gap-2 overflow-hidden">
          <div className="w-7 h-7 rounded bg-[#171f33] flex items-center justify-center flex-shrink-0">
            <Cpu size={14} className="text-[#00daf3]" />
          </div>
          {!isCollapsed && (
            <div className="overflow-hidden whitespace-nowrap">
              <div className="text-[10px] font-mono font-bold text-[#dae2fd] uppercase truncate">
                {hardware || 'CPU Fallback (14T)'}
              </div>
              <div className="text-[9px] font-mono text-[#00f5a0] flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-[#00f5a0] animate-pulse" />
                <span>INFERENCE ENGINE ONLINE</span>
              </div>
            </div>
          )}
        </div>

        {/* Operator Badge */}
        {!isCollapsed && (
          <div className="px-2 py-1.5 flex justify-between items-center text-[10px] font-mono text-[#849588]">
            <span>{activeStation === 'fill' ? 'STATION: LINE-05 (FILL)' : 'STATION: LINE-04 (CAP)'}</span>
            <span className="text-[#00daf3]">OP #4092</span>
          </div>
        )}
      </div>
    </aside>
  );
}
