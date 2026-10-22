import React from 'react';
import { Check, AlertOctagon, AlertTriangle, Eye, Activity } from 'lucide-react';

export default function ConveyorFilmstrip({ inspectionHistory, onSelectBottle }) {
  if (!inspectionHistory || inspectionHistory.length === 0) {
    return (
      <section className="bg-[#0b1326] dark:bg-[#060e20] light:bg-[#ffffff] border border-[#1e2e48] light:border-[#e2e8f0] p-3 rounded shadow-sm">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Activity size={14} className="text-[#00f5a0]" />
            <h3 className="text-xs font-mono font-bold tracking-wider uppercase text-[#dae2fd]">
              Real-Time Conveyor Inspection Queue
            </h3>
          </div>
          <span className="text-[10px] font-mono text-[#849588]">0 UNITS IN QUEUE</span>
        </div>
        <div className="h-20 flex items-center justify-center text-xs font-mono text-[#64748b] bg-[#050811] rounded border border-[#1e2e48]/50">
          Awaiting conveyor stream... Run conveyor or click simulate to populate queue.
        </div>
      </section>
    );
  }

  return (
    <section className="bg-[#0b1326] dark:bg-[#060e20] light:bg-[#ffffff] border border-[#1e2e48] light:border-[#e2e8f0] p-3 rounded shadow-sm flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity size={14} className="text-[#00f5a0]" />
          <h3 className="text-xs font-mono font-bold tracking-wider uppercase text-[#dae2fd] light:text-[#0f172a]">
            Real-Time Conveyor Inspection Queue
          </h3>
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono text-[#849588]">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#00f5a0]" /> PASS
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#ff3b30]" /> REJECT
          </span>
          <span className="text-[#00daf3] font-bold">{inspectionHistory.length} UNITS LOGGED</span>
        </div>
      </div>

      {/* Horizontal Rolling Filmstrip */}
      <div className="flex gap-2.5 overflow-x-auto pb-1.5 custom-scrollbar select-none">
        {inspectionHistory.map((item, index) => {
          const isPass = item.global_verdict === 'PASS';
          const isReject = item.global_verdict === 'REJECT';
          const isUncertain = item.global_verdict === 'UNCERTAIN';
          const frontView = item.views?.camera_front;
          const thumbnail = frontView?.annotated_image_base64;

          return (
            <div
              key={item.bottle_id || index}
              onClick={() => onSelectBottle && onSelectBottle(item.bottle_id)}
              className={`flex-shrink-0 w-36 p-1.5 rounded bg-[#050811] border transition-all duration-150 cursor-pointer hover:scale-[1.02] hover:shadow-lg flex flex-col gap-1.5
                ${isPass 
                  ? 'border-[#1e2e48] hover:border-[#00f5a0]/60' 
                  : isReject 
                    ? 'border-[#ff3b30] shadow-[0_0_8px_rgba(255,59,48,0.3)] hover:border-[#ff817b]' 
                    : 'border-[#f59e0b] hover:border-[#fbbf24]'}`
              }
              title={`Click to inspect ${item.bottle_id}`}
            >
              {/* Header: ID and Timestamp */}
              <div className="flex justify-between items-center text-[10px] font-mono">
                <span className={`font-bold truncate max-w-[70px] ${isReject ? 'text-[#ff817b]' : 'text-[#dae2fd]'}`}>
                  {item.bottle_id}
                </span>
                <span className="text-[#849588]">
                  {item.timestamp ? new Date(item.timestamp).toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' }) : 'NOW'}
                </span>
              </div>

              {/* Center Thumbnail */}
              <div className="h-16 w-full rounded bg-black border border-[#1e2e48] flex items-center justify-center overflow-hidden relative">
                {thumbnail ? (
                  <img src={thumbnail} alt={item.bottle_id} className="w-full h-full object-cover" />
                ) : (
                  <div className="flex items-center justify-center">
                    {isPass && <Check size={20} className="text-[#00f5a0]/80" />}
                    {isReject && <AlertOctagon size={20} className="text-[#ff3b30]" />}
                    {isUncertain && <AlertTriangle size={20} className="text-[#f59e0b]" />}
                  </div>
                )}
                {/* Overlay hover icon */}
                <div className="absolute inset-0 bg-black/50 opacity-0 hover:opacity-100 flex items-center justify-center transition-opacity">
                  <Eye size={16} className="text-[#00daf3]" />
                </div>
              </div>

              {/* Bottom Status Pill */}
              <div className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold text-center uppercase truncate
                ${isPass 
                  ? 'bg-[#00f5a0]/15 text-[#00f5a0] border border-[#00f5a0]/30' 
                  : isReject 
                    ? 'bg-[#ff3b30] text-white' 
                    : 'bg-[#f59e0b]/20 text-[#f59e0b] border border-[#f59e0b]/40'}`
              }>
                {isReject ? (item.primary_defect || 'REJECT') : item.global_verdict}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
