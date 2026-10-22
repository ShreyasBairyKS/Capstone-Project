import React from 'react';
import { Camera, ShieldAlert, ShieldCheck, AlertTriangle, Crosshair, Sparkles } from 'lucide-react';

export default function MultiCameraFeed({ latestInspection, onInspectBottle }) {
  const views = latestInspection?.views || {};
  const globalVerdict = latestInspection?.global_verdict || 'STANDBY';
  const isCompliant = latestInspection?.is_compliant ?? true;
  const primaryDefect = latestInspection?.primary_defect;
  const decisionReason = latestInspection?.decision_reason || 'Conveyor ready. Awaiting bottle inspection trigger.';
  const bottleId = latestInspection?.bottle_id || 'STANDBY_MODE';
  const latency = latestInspection?.total_latency_ms;

  const cameraKeys = Object.keys(views);
  const displayCams = cameraKeys.length > 0 ? cameraKeys : ['camera_front', 'camera_rear'];

  return (
    <section className="bg-[#0b1326] dark:bg-[#060e20] light:bg-[#ffffff] border border-[#1e2e48] light:border-[#e2e8f0] rounded overflow-hidden shadow-sm flex flex-col">
      {/* Global Decision Telemetry Banner */}
      <div className={`p-3.5 border-b flex flex-wrap items-center justify-between gap-3 transition-colors duration-200
        ${globalVerdict === 'PASS' 
          ? 'bg-[#00f5a0]/10 border-[#00f5a0]/30 text-[#00f5a0]' 
          : globalVerdict === 'REJECT' 
            ? 'bg-[#ff3b30]/15 border-[#ff3b30]/40 text-[#ff817b]' 
            : globalVerdict === 'UNCERTAIN'
              ? 'bg-[#f59e0b]/15 border-[#f59e0b]/40 text-[#f59e0b]'
              : 'bg-[#171f33] border-[#1e2e48] text-[#dae2fd]'}`
      }>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            {globalVerdict === 'PASS' && <ShieldCheck size={28} className="text-[#00f5a0]" />}
            {globalVerdict === 'REJECT' && <ShieldAlert size={28} className="text-[#ff3b30]" />}
            {globalVerdict === 'UNCERTAIN' && <AlertTriangle size={28} className="text-[#f59e0b]" />}
            {globalVerdict === 'STANDBY' && <Crosshair size={28} className="text-[#00daf3]" />}
            
            <div>
              <div className="flex items-center gap-2">
                <span className="font-['Outfit'] font-bold text-lg tracking-tight uppercase">
                  {globalVerdict === 'PASS' ? 'PASS - NOMINAL TOLERANCE' : globalVerdict === 'REJECT' ? `REJECT - ${primaryDefect?.toUpperCase() || 'DEFECT DETECTED'}` : globalVerdict}
                </span>
                <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-black/40 border border-current">
                  {bottleId}
                </span>
                {bottleId !== 'STANDBY_MODE' && onInspectBottle && (
                  <button
                    onClick={() => onInspectBottle(bottleId)}
                    className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#00daf3]/20 hover:bg-[#00daf3]/30 text-[#00daf3] border border-[#00daf3]/50 transition-colors flex items-center gap-1 cursor-pointer"
                    title="Open forensic contour replay in S2"
                  >
                    <span>REPLAY INSPECT</span>
                  </button>
                )}
              </div>
              <p className="text-xs text-[#94a3b8] mt-0.5">
                {decisionReason}
              </p>
            </div>
          </div>
        </div>

        {/* Telemetry quick status chips */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="text-right">
            <div className="text-[10px] text-[#849588]">EJECTION ACTUATOR</div>
            <div className={`font-bold ${latestInspection?.trigger_reject_actuator ? 'text-[#ff3b30]' : 'text-[#00f5a0]'}`}>
              {latestInspection?.trigger_reject_actuator ? 'PNEUMATIC EJECT FIRED' : 'PASSED DOWNSTREAM'}
            </div>
          </div>
          <div className="text-right border-l border-[#1e2e48] pl-3">
            <div className="text-[10px] text-[#849588]">CYCLE TIME</div>
            <div className="font-bold text-[#00daf3]">
              {latency ? `${latency.toFixed(1)} ms` : '—'}
            </div>
          </div>
        </div>
      </div>

      {/* Dual-Camera Machine Vision Viewports */}
      <div className="p-3 grid grid-cols-1 md:grid-cols-2 gap-3">
        {displayCams.map((camId) => {
          const viewData = views[camId];
          const hasImage = !!viewData?.annotated_image_base64;
          const verdict = viewData?.verdict || 'STANDBY';
          const conf = viewData?.primary_confidence;
          const defect = viewData?.defect_name;

          return (
            <div 
              key={camId} 
              className="bg-[#050811] rounded border border-[#1e2e48] overflow-hidden flex flex-col relative group"
            >
              {/* Viewport Top Bar */}
              <div className="h-8 bg-[#0b111e] border-b border-[#1e2e48] px-3 flex justify-between items-center text-xs font-mono">
                <div className="flex items-center gap-2 text-[#dae2fd]">
                  <Camera size={14} className="text-[#00daf3]" />
                  <span className="font-bold uppercase tracking-wider">{camId.replace('_', ' ')}</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-[#849588]">
                  <span>POLARIZED 120 FPS</span>
                  <span className="px-1.5 py-0.5 rounded bg-[#171f33] text-[#00f5a0]">
                    {viewData?.detection_count !== undefined ? `${viewData.detection_count} obj` : 'READY'}
                  </span>
                </div>
              </div>

              {/* Viewport Image Canvas Container */}
              <div className="w-full aspect-[4/3] bg-black relative flex items-center justify-center overflow-hidden">
                {/* Crosshair Grid Overlay */}
                <div className="absolute inset-0 pointer-events-none opacity-20 bg-[linear-gradient(to_right,#1e2e48_1px,transparent_1px),linear-gradient(to_bottom,#1e2e48_1px,transparent_1px)] bg-[size:32px_32px]" />
                
                {hasImage ? (
                  <>
                    <img 
                      src={viewData.annotated_image_base64} 
                      alt={camId} 
                      className="w-full h-full object-contain relative z-10" 
                    />

                    {/* Camera Angle Verdict Badge */}
                    <div className={`absolute top-2.5 left-2.5 z-20 px-2 py-1 rounded text-xs font-mono font-bold backdrop-blur-md border shadow-md
                      ${verdict === 'PASS' 
                        ? 'bg-[#00f5a0]/90 text-black border-[#00f5a0]' 
                        : verdict === 'REJECT' 
                          ? 'bg-[#ff3b30]/90 text-white border-[#ff3b30]' 
                          : 'bg-[#f59e0b]/90 text-black border-[#f59e0b]'}`
                    }>
                      {verdict === 'REJECT' ? `REJECT: ${defect || 'DEFECT'}` : verdict}
                      {conf ? ` (${(conf * 100).toFixed(1)}%)` : ''}
                    </div>

                    {/* Timestamp & Coordinate Watermark */}
                    <div className="absolute bottom-2 left-2 z-20 text-[9px] font-mono text-[#00daf3]/80 bg-black/60 px-1.5 py-0.5 rounded border border-[#1e2e48]">
                      SEAL RIM RETICLE • 28.1mm ±0.04mm
                    </div>
                  </>
                ) : (
                  <div className="flex flex-col items-center gap-2 text-[#849588] p-6 text-center z-10">
                    <Crosshair size={36} className="text-[#00daf3]/40 animate-pulse" />
                    <span className="text-xs font-mono">Awaiting Conveyor Strobe Trigger...</span>
                    <span className="text-[10px] text-[#64748b]">Press "SIMULATE" to inspect a bottle</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
