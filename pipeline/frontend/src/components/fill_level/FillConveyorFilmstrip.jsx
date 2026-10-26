import React from 'react';
import { Layers, CheckCircle2, AlertOctagon, AlertTriangle, Droplets, ChevronRight } from 'lucide-react';

export default function FillConveyorFilmstrip({ inspectionHistory = [], onSelectBottle }) {
  if (!inspectionHistory || inspectionHistory.length === 0) {
    return (
      <div className="p-3 rounded bg-[#0b1326] border border-[#1e2e48] flex items-center justify-between text-[#849588] text-xs font-mono">
        <div className="flex items-center gap-2">
          <Layers size={14} className="text-[#00daf3]" />
          <span>STATION 2 CONVEYOR FILMSTRIP: Awaiting initial bottle inspections...</span>
        </div>
        <span className="text-[10px] text-[#849588]">QUEUE CAP: 20 SPECIMENS</span>
      </div>
    );
  }

  return (
    <div className="p-3 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-col gap-2 shadow-sm">
      <div className="flex items-center justify-between border-b border-[#1e2e48] pb-1.5">
        <div className="flex items-center gap-2 text-xs font-mono font-bold text-[#dae2fd]">
          <Layers size={14} className="text-[#00daf3]" />
          <span>CONVEYOR SPECIMEN QUEUE (LAST {inspectionHistory.length})</span>
        </div>
        <div className="text-[10px] font-mono text-[#849588]">
          CLICK SPECIMEN CARD FOR S2 REPLAY DEEP-DIVE
        </div>
      </div>

      {/* Horizontal Scrolling Queue */}
      <div className="flex items-center gap-2.5 overflow-x-auto pb-1 pt-0.5 scrollbar-thin scrollbar-thumb-[#1e2e48]">
        {inspectionHistory.map((b, idx) => {
          const isPass = b.verdict === 'PASS';
          const isReject = b.verdict === 'REJECT';
          const isUncertain = b.verdict === 'UNCERTAIN';
          const ratio = b.fill_ratio != null ? (b.fill_ratio * 100).toFixed(1) : '--';
          const fillStatus = b.fill_status || 'normal';

          return (
            <div
              key={b.bottle_id || idx}
              onClick={() => onSelectBottle?.(b.bottle_id)}
              className={`flex-shrink-0 w-44 p-2 rounded bg-[#070d1c] border cursor-pointer transition-all duration-150 hover:-translate-y-0.5 hover:shadow-md
                ${isPass ? 'border-[#00f5a0]/40 hover:border-[#00f5a0] hover:shadow-[0_0_8px_rgba(0,245,160,0.15)]' : ''}
                ${isReject ? 'border-[#ff3b30]/40 hover:border-[#ff3b30] hover:shadow-[0_0_8px_rgba(255,59,48,0.15)]' : ''}
                ${isUncertain ? 'border-[#f59e0b]/40 hover:border-[#f59e0b] hover:shadow-[0_0_8px_rgba(245,158,11,0.15)]' : ''}`
              }
            >
              {/* Card Header */}
              <div className="flex items-center justify-between text-[10px] font-mono mb-1">
                <span className="font-bold text-[#dae2fd] truncate max-w-[110px]" title={b.bottle_id}>
                  {b.bottle_id ? b.bottle_id.slice(-10) : `SPEC-${idx + 1}`}
                </span>
                {isPass && <CheckCircle2 size={12} className="text-[#00f5a0] flex-shrink-0" />}
                {isReject && <AlertOctagon size={12} className="text-[#ff3b30] flex-shrink-0" />}
                {isUncertain && <AlertTriangle size={12} className="text-[#f59e0b] flex-shrink-0" />}
              </div>

              {/* Thumbnail or Ratio Display */}
              <div className="h-16 rounded bg-[#050811] border border-[#1e2e48] overflow-hidden flex items-center justify-center relative mb-1.5">
                {b.annotated_image_base64 ? (
                  <img
                    src={b.annotated_image_base64}
                    alt={b.bottle_id}
                    className="h-full w-full object-contain"
                  />
                ) : (
                  <div className="flex flex-col items-center justify-center text-[#849588]">
                    <Droplets size={20} className="text-[#00daf3]/50" />
                    <span className="text-[9px] font-mono mt-0.5">{ratio}%</span>
                  </div>
                )}

                {/* Ratio badge */}
                <span className={`absolute bottom-1 right-1 px-1 py-0.2 rounded text-[9px] font-mono font-bold
                  ${isPass ? 'bg-[#00f5a0]/90 text-black' : ''}
                  ${isReject ? 'bg-[#ff3b30]/90 text-white' : ''}
                  ${isUncertain ? 'bg-[#fbbf24]/90 text-black' : ''}`
                }>
                  {ratio}%
                </span>
              </div>

              {/* Status footer */}
              <div className="flex items-center justify-between text-[9px] font-mono text-[#849588]">
                <span className="uppercase font-semibold truncate">{fillStatus}</span>
                <span className="text-[#00daf3] flex items-center">
                  VIEW <ChevronRight size={10} />
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
