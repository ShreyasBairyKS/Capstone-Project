import React from 'react';
import { ShieldCheck, Droplets, Loader2, Cpu } from 'lucide-react';
import { useStation } from '../../context/StationContext';

export default function StationSwitcher({ isCollapsed = false, compact = false }) {
  const { activeStation, switchStation, isSwapping, swapMessage, capHealth, fillHealth } = useStation();

  if (compact) {
    return (
      <div className="flex items-center bg-[#091122] border border-[#1e2e48] rounded p-0.5 select-none shadow-sm">
        <button
          onClick={() => switchStation('cap')}
          disabled={isSwapping}
          title="Station 1: Cap Defect Inspection (3-Camera Array)"
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono font-bold transition-all duration-150
            ${activeStation === 'cap'
              ? 'bg-[#17253d] text-[#00f5a0] shadow-[0_0_8px_rgba(0,245,160,0.25)] border border-[#00f5a0]/40'
              : 'text-[#849588] hover:text-[#dae2fd] hover:bg-[#131b2e]'}`
          }
        >
          <ShieldCheck size={13} className={activeStation === 'cap' ? 'text-[#00f5a0]' : ''} />
          <span>STATION 1 (CAP)</span>
        </button>

        <button
          onClick={() => switchStation('fill')}
          disabled={isSwapping}
          title="Station 2: Fill Level Inspection (Profile Camera)"
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono font-bold transition-all duration-150
            ${activeStation === 'fill'
              ? 'bg-[#17253d] text-[#00daf3] shadow-[0_0_8px_rgba(0,218,243,0.25)] border border-[#00daf3]/40'
              : 'text-[#849588] hover:text-[#dae2fd] hover:bg-[#131b2e]'}`
          }
        >
          <Droplets size={13} className={activeStation === 'fill' ? 'text-[#00daf3]' : ''} />
          <span>STATION 2 (FILL)</span>
        </button>

        {isSwapping && (
          <div className="flex items-center gap-1.5 px-2 text-[10px] font-mono text-[#fbbf24] animate-pulse">
            <Loader2 size={12} className="animate-spin" />
            <span className="hidden md:inline">SWAPPING VRAM...</span>
          </div>
        )}
      </div>
    );
  }

  // Sidebar Full/Collapsed Representation
  if (isCollapsed) {
    return (
      <div className="flex flex-col gap-1 items-center px-1 py-2 border-y border-[#1e2e48] bg-[#070e1e]">
        <button
          onClick={() => switchStation('cap')}
          disabled={isSwapping}
          title="Station 1: Cap Inspection"
          className={`w-9 h-9 rounded flex items-center justify-center transition-all duration-150 relative
            ${activeStation === 'cap'
              ? 'bg-[#17253d] text-[#00f5a0] border border-[#00f5a0]/40 shadow-[0_0_10px_rgba(0,245,160,0.3)]'
              : 'text-[#849588] hover:text-[#dae2fd] hover:bg-[#131b2e]'}`
          }
        >
          <ShieldCheck size={18} />
          <span className={`absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full ${activeStation === 'cap' ? 'bg-[#00f5a0]' : 'bg-transparent'}`} />
        </button>

        <button
          onClick={() => switchStation('fill')}
          disabled={isSwapping}
          title="Station 2: Fill Level Inspection"
          className={`w-9 h-9 rounded flex items-center justify-center transition-all duration-150 relative
            ${activeStation === 'fill'
              ? 'bg-[#17253d] text-[#00daf3] border border-[#00daf3]/40 shadow-[0_0_10px_rgba(0,218,243,0.3)]'
              : 'text-[#849588] hover:text-[#dae2fd] hover:bg-[#131b2e]'}`
          }
        >
          <Droplets size={18} />
          <span className={`absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full ${activeStation === 'fill' ? 'bg-[#00daf3]' : 'bg-transparent'}`} />
        </button>
      </div>
    );
  }

  return (
    <div className="p-2 rounded bg-[#070e1e] border border-[#1e2e48] flex flex-col gap-1.5 select-none shadow-inner">
      <div className="flex items-center justify-between px-1 text-[10px] font-mono font-bold tracking-wider text-[#849588]">
        <span>INSPECTION STATION</span>
        {isSwapping ? (
          <span className="text-[#fbbf24] flex items-center gap-1 animate-pulse">
            <Loader2 size={10} className="animate-spin" />
            SWAPPING VRAM
          </span>
        ) : (
          <span className="text-[#00daf3] font-semibold">HOT-SWAP READY</span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-1.5">
        {/* Station 1 Card */}
        <button
          onClick={() => switchStation('cap')}
          disabled={isSwapping}
          className={`flex flex-col gap-1 p-2 rounded text-left transition-all duration-150 border relative
            ${activeStation === 'cap'
              ? 'bg-[#17253d] border-[#00f5a0]/50 text-[#dae2fd] shadow-[0_0_12px_rgba(0,245,160,0.15)]'
              : 'bg-[#0c1426] border-[#1e2e48] text-[#849588] hover:bg-[#131b2e] hover:text-[#dae2fd]'}`
          }
        >
          <div className="flex items-center justify-between">
            <ShieldCheck size={14} className={activeStation === 'cap' ? 'text-[#00f5a0]' : 'text-[#849588]'} />
            <span className={`w-1.5 h-1.5 rounded-full ${activeStation === 'cap' ? 'bg-[#00f5a0] animate-pulse' : 'bg-transparent'}`} />
          </div>
          <div className="text-[11px] font-mono font-bold tracking-tight">STATION 1</div>
          <div className="text-[9px] font-mono text-[#849588]">CAP DEFECT</div>
        </button>

        {/* Station 2 Card */}
        <button
          onClick={() => switchStation('fill')}
          disabled={isSwapping}
          className={`flex flex-col gap-1 p-2 rounded text-left transition-all duration-150 border relative
            ${activeStation === 'fill'
              ? 'bg-[#17253d] border-[#00daf3]/50 text-[#dae2fd] shadow-[0_0_12px_rgba(0,218,243,0.15)]'
              : 'bg-[#0c1426] border-[#1e2e48] text-[#849588] hover:bg-[#131b2e] hover:text-[#dae2fd]'}`
          }
        >
          <div className="flex items-center justify-between">
            <Droplets size={14} className={activeStation === 'fill' ? 'text-[#00daf3]' : 'text-[#849588]'} />
            <span className={`w-1.5 h-1.5 rounded-full ${activeStation === 'fill' ? 'bg-[#00daf3] animate-pulse' : 'bg-transparent'}`} />
          </div>
          <div className="text-[11px] font-mono font-bold tracking-tight">STATION 2</div>
          <div className="text-[9px] font-mono text-[#849588]">FILL LEVEL</div>
        </button>
      </div>

      {isSwapping && (
        <div className="text-[9px] font-mono text-[#fbbf24] px-1 italic truncate">
          {swapMessage || 'Switching active GPU model...'}
        </div>
      )}
    </div>
  );
}
