import React from 'react';
import { useStation } from '../../context/StationContext';
import { Camera, Eye, AlertTriangle, AlertOctagon, CheckCircle2 } from 'lucide-react';

export default function FillSingleFeed({ latestInspection, onInspectBottle }) {
  const { fillHealth } = useStation();
  const underRatio = fillHealth?.underfill_threshold ?? 0.40;
  const overRatio = fillHealth?.overfill_threshold ?? 0.85;
  const underPct = Math.round(underRatio * 100);
  const overPct = Math.round(overRatio * 100);

  const bottleId = latestInspection?.bottle_id || 'AWAITING INSPECTION';
  const timestamp = latestInspection?.timestamp || '--';
  const verdict = latestInspection?.verdict || 'IDLE';
  const fillStatus = latestInspection?.fill_status || 'normal';
  const fillRatio = latestInspection?.fill_ratio != null ? (latestInspection.fill_ratio * 100).toFixed(1) : '--';
  const latency = latestInspection?.total_latency_ms != null ? latestInspection.total_latency_ms.toFixed(2) : '--';
  const reason = latestInspection?.decision_reason || 'System standing by on Station 2.';

  // Image source: if showOverlay, use annotated_image_base64, else raw image or placeholder
  const imageSrc = latestInspection?.annotated_image_base64;

  const isPass = verdict === 'PASS';
  const isReject = verdict === 'REJECT';
  const isUncertain = verdict === 'UNCERTAIN';

  return (
    <div className="p-3.5 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-col gap-3 shadow-md">
      {/* Top Header / Camera Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#1e2e48] pb-2.5">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded bg-[#17253d] border border-[#00daf3]/40 text-[#00daf3]">
            <Camera size={16} />
          </div>
          <div>
            <div className="text-xs font-mono font-bold text-[#dae2fd] flex items-center gap-2">
              <span>CAM-04: PROFILE LIQUID LEVEL INSPECTION</span>
              <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-[#00daf3]/10 text-[#00daf3] border border-[#00daf3]/30">
                BATCH=1
              </span>
            </div>
            <div className="text-[10px] font-mono text-[#849588]">
              MONOCHROME HIGH-SPEED SENSOR &bull; 640&times;640 RES &bull; TENSORRT ENGINE
            </div>
          </div>
        </div>

        {/* Right Controls */}
        <div className="flex items-center gap-2">
          {latestInspection?.bottle_id && (
            <button
              onClick={() => onInspectBottle?.(latestInspection.bottle_id)}
              className="px-2.5 py-1 rounded text-xs font-mono font-semibold bg-[#17253d] text-[#dae2fd] border border-[#1e2e48] hover:border-[#00daf3] hover:text-[#00daf3] transition-colors flex items-center gap-1.5"
            >
              <Eye size={13} />
              <span>REPLAY DEEP-DIVE</span>
            </button>
          )}
        </div>
      </div>

      {/* Main Machine Vision Canvas & Live Readout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5">
        {/* Left: Viewport (8 Cols) */}
        <div className="lg:col-span-8 relative rounded bg-[#060a14] border border-[#1e2e48] overflow-hidden min-h-[380px] flex items-center justify-center">
          {imageSrc ? (
            <div className="relative w-full h-full flex items-center justify-center p-2">
              <img
                src={imageSrc}
                alt="Station 2 Profile Inspection"
                className="max-h-[460px] w-auto object-contain rounded border border-[#1e2e48]/50 shadow-lg"
              />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center text-[#849588] p-8 text-center gap-2">
              <Camera size={36} className="text-[#1e2e48]" />
              <div className="text-xs font-mono font-bold text-[#dae2fd]">AWAITING PROFILE CAMERA FRAME</div>
              <div className="text-[11px] font-mono text-[#849588] max-w-sm">
                Press &quot;SIMULATE CYCLE&quot; in the top header or run continuous conveyor stream to evaluate bottle fill level.
              </div>
            </div>
          )}

          {/* Viewport Floating Status Pill */}
          {latestInspection && (
            <div className="absolute top-3 left-3 flex items-center gap-2">
              <span className={`px-2.5 py-1 rounded text-xs font-mono font-bold border shadow-md flex items-center gap-1.5
                ${isPass ? 'bg-[#00f5a0]/20 text-[#00f5a0] border-[#00f5a0]/50 shadow-[0_0_10px_rgba(0,245,160,0.2)]' : ''}
                ${isReject ? 'bg-[#ff3b30]/20 text-[#ff817b] border-[#ff3b30]/50 shadow-[0_0_10px_rgba(255,59,48,0.2)]' : ''}
                ${isUncertain ? 'bg-[#f59e0b]/20 text-[#fbbf24] border-[#f59e0b]/50 shadow-[0_0_10px_rgba(245,158,11,0.2)]' : ''}`
              }>
                {isPass && <CheckCircle2 size={13} />}
                {isReject && <AlertOctagon size={13} />}
                {isUncertain && <AlertTriangle size={13} />}
                <span>{verdict} &bull; {fillRatio}%</span>
              </span>
            </div>
          )}

          {/* Calibrated Liquid Ruler Overlay Legend Indicator */}
          <div className="absolute bottom-3 right-3 px-2 py-1 rounded bg-[#060a14]/85 border border-[#1e2e48] text-[10px] font-mono text-[#849588] backdrop-blur-sm">
            <span className="text-[#00daf3] font-bold">RULER:</span> {underPct}% MIN &bull; {overPct}% MAX
          </div>
        </div>

        {/* Right: Real-Time Diagnostic Card (4 Cols) */}
        <div className="lg:col-span-4 flex flex-col justify-between p-3.5 rounded bg-[#080f20] border border-[#1e2e48] gap-3">
          <div className="flex flex-col gap-2.5">
            <div className="text-[10px] font-mono font-bold text-[#849588] tracking-wider uppercase border-b border-[#1e2e48] pb-1.5">
              INSPECTION TELEMETRY MATRIX
            </div>

            {/* Bottle ID & Timestamp */}
            <div className="flex flex-col gap-0.5">
              <div className="text-[10px] font-mono text-[#849588]">BOTTLE SPECIMEN ID</div>
              <div className="text-sm font-mono font-bold text-[#00daf3] truncate">
                {bottleId}
              </div>
              <div className="text-[10px] font-mono text-[#849588]">
                {timestamp}
              </div>
            </div>

            {/* Verdict Display Banner */}
            <div className={`p-2.5 rounded border flex items-center justify-between
              ${isPass ? 'bg-[#00f5a0]/10 border-[#00f5a0]/30 text-[#00f5a0]' : ''}
              ${isReject ? 'bg-[#ff3b30]/10 border-[#ff3b30]/30 text-[#ff817b]' : ''}
              ${isUncertain ? 'bg-[#f59e0b]/10 border-[#f59e0b]/30 text-[#fbbf24]' : ''}
              ${verdict === 'IDLE' ? 'bg-[#131b2e] border-[#1e2e48] text-[#849588]' : ''}`
            }>
              <div className="flex flex-col">
                <span className="text-[10px] font-mono text-[#849588]">QUALITY GATE VERDICT</span>
                <span className="text-base font-mono font-black tracking-tight">{verdict}</span>
              </div>
              <div className="text-right">
                <span className="text-[10px] font-mono text-[#849588]">FILL STATUS</span>
                <div className="text-xs font-mono font-bold uppercase">{fillStatus}</div>
              </div>
            </div>

            {/* Fill Level Metric Gauge */}
            <div className="p-2.5 rounded bg-[#060a14] border border-[#1e2e48] flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-[#849588]">LIQUID VOLUME RATIO:</span>
                <span className="font-bold text-[#00daf3] text-sm">{fillRatio}%</span>
              </div>

              {/* Progress bar with nominal limit ticks */}
              <div className="relative w-full h-3 bg-[#131b2e] rounded overflow-hidden">
                {/* 40% min line */}
                <div className="absolute top-0 bottom-0 w-0.5 bg-[#f59e0b] z-10" style={{ left: `${underPct}%` }} title={`${underPct}% Min Limit`} />
                {/* 85% max line */}
                <div className="absolute top-0 bottom-0 w-0.5 bg-[#ff3b30] z-10" style={{ left: `${overPct}%` }} title={`${overPct}% Max Limit`} />

                {/* Actual Fill Bar */}
                <div
                  className={`h-full transition-all duration-300
                    ${isPass ? 'bg-[#00f5a0]' : ''}
                    ${isReject && (latestInspection?.fill_ratio || 0) > overRatio ? 'bg-[#ff3b30]' : ''}
                    ${isReject && (latestInspection?.fill_ratio || 0) < underRatio ? 'bg-[#f59e0b]' : ''}
                    ${isUncertain ? 'bg-[#fbbf24]' : ''}
                    ${verdict === 'IDLE' ? 'bg-[#1e2e48]' : ''}`
                  }
                  style={{ width: `${Math.min(100, Math.max(0, latestInspection?.fill_ratio ? latestInspection.fill_ratio * 100 : 0))}%` }}
                />
              </div>

              <div className="flex justify-between text-[9px] font-mono text-[#849588]">
                <span>0% EMPTY</span>
                <span className="text-[#f59e0b]">{underPct}% MIN</span>
                <span className="text-[#00f5a0]">NOMINAL</span>
                <span className="text-[#ff3b30]">{overPct}% MAX</span>
                <span>100% OVER</span>
              </div>
            </div>

            {/* Decision Reason */}
            <div className="p-2 rounded bg-[#060a14] border border-[#1e2e48] text-xs font-mono text-[#dae2fd]">
              <div className="text-[10px] text-[#849588] font-bold">DECISION REASON:</div>
              <div className="mt-0.5 text-[11px] leading-relaxed">{reason}</div>
            </div>
          </div>

          {/* Latency Footnote */}
          <div className="pt-2 border-t border-[#1e2e48] flex items-center justify-between text-[10px] font-mono text-[#849588]">
            <span>PROCESSING LATENCY:</span>
            <span className="text-[#00f5a0] font-bold">{latency} ms</span>
          </div>
        </div>
      </div>
    </div>
  );
}
