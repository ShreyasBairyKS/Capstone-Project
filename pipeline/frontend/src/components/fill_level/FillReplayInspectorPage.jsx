import React, { useState, useEffect, useCallback } from 'react';
import { 
  ChevronLeft, 
  ChevronRight, 
  Eye, 
  Layers, 
  Copy, 
  Check, 
  Droplets, 
  Clock, 
  CheckCircle2, 
  AlertOctagon, 
  AlertTriangle,
  BrainCircuit,
  Maximize2
} from 'lucide-react';

export default function FillReplayInspectorPage({ 
  initialBottleId, 
  onSelectBottle, 
  onTriageToAnnotation 
}) {
  const [bottleId, setBottleId] = useState(initialBottleId || '');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState('split'); // 'split', 'raw', 'overlay'

  // Fetch bottle detail
  const fetchBottle = useCallback(async (id) => {
    if (!id) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/fill/bottle/${encodeURIComponent(id)}`);
      if (res.ok) {
        const json = await res.json();
        setData(json);
        setBottleId(id);
      } else {
        console.warn(`Bottle inspection ${id} not found.`);
      }
    } catch (err) {
      console.error('Error fetching fill bottle detail:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch initial bottle or recent bottle if none selected
  useEffect(() => {
    if (initialBottleId) {
      fetchBottle(initialBottleId);
    } else {
      // Find latest bottle from defects endpoint
      fetch('/api/fill/defects?limit=1')
        .then(r => r.json())
        .then(res => {
          const first = res.items?.[0] || res[0];
          if (first?.bottle_id) {
            fetchBottle(first.bottle_id);
          }
        })
        .catch(() => null);
    }
  }, [initialBottleId, fetchBottle]);

  const record = data?.record;
  const adjacent = data?.adjacent || {};
  const rawB64 = data?.raw_image_base64;
  const overlayB64 = data?.annotated_image_base64;

  const handleCopyId = () => {
    if (!bottleId) return;
    navigator.clipboard.writeText(bottleId);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleNavigate = (targetId) => {
    if (!targetId) return;
    setBottleId(targetId);
    onSelectBottle?.(targetId);
    fetchBottle(targetId);
  };

  const isPass = record?.global_verdict === 'PASS';
  const isReject = record?.global_verdict === 'REJECT';
  const isUnc = record?.global_verdict === 'UNCERTAIN';
  const ratio = record?.fill_ratio != null ? (record.fill_ratio * 100).toFixed(1) : '--';

  return (
    <div className="flex flex-col gap-3.5 w-full">
      {/* Top Stepper & Navigation Bar */}
      <div className="p-3.5 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-wrap items-center justify-between gap-3 shadow-sm">
        {/* Left: Stepper Navigation */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleNavigate(adjacent.prev_id)}
            disabled={!adjacent.prev_id || loading}
            className="px-2.5 py-1.5 rounded bg-[#17253d] border border-[#1e2e48] disabled:opacity-40 disabled:cursor-not-allowed hover:border-[#00daf3] text-[#dae2fd] text-xs font-mono font-semibold flex items-center gap-1 transition-colors"
            title="Inspect Previous Bottle"
          >
            <ChevronLeft size={16} />
            <span className="hidden sm:inline">PREV SPECIMEN</span>
          </button>

          <div className="flex items-center gap-2 px-3 py-1 rounded bg-[#070e1e] border border-[#1e2e48]">
            <Droplets size={14} className="text-[#00daf3]" />
            <span className="text-xs font-mono font-bold text-[#dae2fd]">
              {bottleId || 'NO SPECIMEN SELECTED'}
            </span>
            {bottleId && (
              <button
                onClick={handleCopyId}
                className="text-[#849588] hover:text-[#dae2fd] p-0.5 rounded"
                title="Copy Bottle ID"
              >
                {copied ? <Check size={12} className="text-[#00f5a0]" /> : <Copy size={12} />}
              </button>
            )}
          </div>

          <button
            onClick={() => handleNavigate(adjacent.next_id)}
            disabled={!adjacent.next_id || loading}
            className="px-2.5 py-1.5 rounded bg-[#17253d] border border-[#1e2e48] disabled:opacity-40 disabled:cursor-not-allowed hover:border-[#00daf3] text-[#dae2fd] text-xs font-mono font-semibold flex items-center gap-1 transition-colors"
            title="Inspect Next Bottle"
          >
            <span className="hidden sm:inline">NEXT SPECIMEN</span>
            <ChevronRight size={16} />
          </button>
        </div>

        {/* View Mode Segmented Control */}
        <div className="flex items-center bg-[#070e1e] border border-[#1e2e48] rounded p-0.5">
          <button
            onClick={() => setViewMode('split')}
            className={`px-2.5 py-1 rounded text-xs font-mono font-semibold transition-colors
              ${viewMode === 'split' ? 'bg-[#17253d] text-[#00daf3] border border-[#00daf3]/40' : 'text-[#849588] hover:text-[#dae2fd]'}`
            }
          >
            SPLIT RAW / OVERLAY
          </button>
          <button
            onClick={() => setViewMode('overlay')}
            className={`px-2.5 py-1 rounded text-xs font-mono font-semibold transition-colors
              ${viewMode === 'overlay' ? 'bg-[#17253d] text-[#00daf3] border border-[#00daf3]/40' : 'text-[#849588] hover:text-[#dae2fd]'}`
            }
          >
            OVERLAY ONLY
          </button>
          <button
            onClick={() => setViewMode('raw')}
            className={`px-2.5 py-1 rounded text-xs font-mono font-semibold transition-colors
              ${viewMode === 'raw' ? 'bg-[#17253d] text-[#00daf3] border border-[#00daf3]/40' : 'text-[#849588] hover:text-[#dae2fd]'}`
            }
          >
            RAW SENSOR ONLY
          </button>
        </div>

        {/* Right: Active Learning Triage Action */}
        {onTriageToAnnotation && (
          <button
            onClick={() => onTriageToAnnotation(bottleId)}
            className="px-3 py-1.5 rounded bg-[#17253d] border border-[#fbbf24]/40 text-[#fbbf24] hover:bg-[#25392b] text-xs font-mono font-bold flex items-center gap-1.5 transition-colors"
          >
            <BrainCircuit size={13} />
            <span>CALIBRATE IN S3</span>
          </button>
        )}
      </div>

      {/* Main Viewport & Telemetry Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-3.5">
        {/* Viewport Canvas (8 Cols) */}
        <div className="xl:col-span-8 flex flex-col gap-3">
          <div className="p-3 rounded bg-[#0b1326] border border-[#1e2e48] shadow-sm">
            <div className="flex items-center justify-between pb-2 border-b border-[#1e2e48] text-xs font-mono text-[#849588]">
              <span className="font-bold text-[#dae2fd] flex items-center gap-1.5">
                <Layers size={14} className="text-[#00daf3]" />
                HIGH-RESOLUTION MACHINE VISION PROFILE CANVAS
              </span>
              <span>CALIBRATED WATER MENISCUS & RULER</span>
            </div>

            {/* Split View Container */}
            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
              {/* Left View: Raw Capture */}
              {(viewMode === 'split' || viewMode === 'raw') && (
                <div className={`relative rounded bg-[#060a14] border border-[#1e2e48] p-2 flex flex-col items-center justify-center min-h-[380px]
                  ${viewMode === 'raw' ? 'md:col-span-2' : ''}`
                }>
                  <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-[#070e1e]/80 border border-[#1e2e48] text-[10px] font-mono text-[#849588] z-10">
                    RAW CAPTURE (CLEAN TRAINING-READY)
                  </div>
                  {rawB64 ? (
                    <img 
                      src={rawB64} 
                      alt="Raw Sensor Capture"
                      className="max-h-[420px] w-auto object-contain rounded"
                    />
                  ) : (
                    <div className="text-[#849588] text-xs font-mono">No raw capture available</div>
                  )}
                </div>
              )}

              {/* Right View: Dynamic HUD Overlay */}
              {(viewMode === 'split' || viewMode === 'overlay') && (
                <div className={`relative rounded bg-[#060a14] border border-[#1e2e48] p-2 flex flex-col items-center justify-center min-h-[380px]
                  ${viewMode === 'overlay' ? 'md:col-span-2' : ''}`
                }>
                  <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-[#070e1e]/80 border border-[#00daf3]/40 text-[10px] font-mono text-[#00daf3] z-10">
                    DYNAMIC HUD RECONSTRUCTION
                  </div>
                  {overlayB64 ? (
                    <img 
                      src={overlayB64} 
                      alt="Dynamic Overlay Reconstructed"
                      className="max-h-[420px] w-auto object-contain rounded shadow-[0_0_15px_rgba(0,218,243,0.15)]"
                    />
                  ) : (
                    <div className="text-[#849588] text-xs font-mono">No overlay available</div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Telemetry Breakdown (4 Cols) */}
        <div className="xl:col-span-4 flex flex-col gap-3">
          {/* Decision Card */}
          <div className="p-3.5 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-col gap-2.5 shadow-sm">
            <div className="text-[10px] font-mono font-bold text-[#849588] tracking-wider uppercase border-b border-[#1e2e48] pb-1.5">
              SPECIMEN VERDICT TELEMETRY
            </div>

            <div className={`p-3 rounded border flex items-center justify-between
              ${isPass ? 'bg-[#00f5a0]/10 border-[#00f5a0]/30 text-[#00f5a0]' : ''}
              ${isReject ? 'bg-[#ff3b30]/10 border-[#ff3b30]/30 text-[#ff817b]' : ''}
              ${isUnc ? 'bg-[#f59e0b]/10 border-[#f59e0b]/30 text-[#fbbf24]' : ''}
              ${!record ? 'bg-[#131b2e] border-[#1e2e48] text-[#849588]' : ''}`
            }>
              <div>
                <span className="text-[10px] font-mono text-[#849588]">DECISION</span>
                <div className="text-lg font-mono font-black">{record?.global_verdict || '--'}</div>
              </div>
              <div className="text-right">
                <span className="text-[10px] font-mono text-[#849588]">CLASSIFICATION</span>
                <div className="text-xs font-mono font-bold uppercase">{record?.fill_status || '--'}</div>
              </div>
            </div>

            {/* Measurements Detail */}
            <div className="flex flex-col gap-1.5 text-xs font-mono pt-1">
              <div className="flex justify-between py-1 border-b border-[#1e2e48]/50">
                <span className="text-[#849588]">LIQUID RATIO:</span>
                <span className="font-bold text-[#00daf3]">{ratio}%</span>
              </div>
              <div className="flex justify-between py-1 border-b border-[#1e2e48]/50">
                <span className="text-[#849588]">WATER MENISCUS Y:</span>
                <span className="font-bold text-[#dae2fd]">{record?.water_y != null ? `${record.water_y} px` : '--'}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-[#1e2e48]/50">
                <span className="text-[#849588]">CONTAINER HEIGHT:</span>
                <span className="font-bold text-[#dae2fd]">{record?.bottle_h != null ? `${record.bottle_h} px` : '--'}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-[#1e2e48]/50">
                <span className="text-[#849588]">WATER DETECTOR CONF:</span>
                <span className="font-bold text-[#00f5a0]">{record?.water_conf != null ? `${(record.water_conf * 100).toFixed(1)}%` : '--'}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-[#1e2e48]/50">
                <span className="text-[#849588]">BOTTLE DETECTOR CONF:</span>
                <span className="font-bold text-[#00f5a0]">{record?.bottle_conf != null ? `${(record.bottle_conf * 100).toFixed(1)}%` : '--'}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-[#1e2e48]/50">
                <span className="text-[#849588]">PNEUMATIC ACTUATOR:</span>
                <span className={`font-bold ${record?.trigger_actuator ? 'text-[#ff817b]' : 'text-[#849588]'}`}>
                  {record?.trigger_actuator ? 'FIRED (EJECT)' : 'ARMED (IN-LINE)'}
                </span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-[#849588]">CYCLE LATENCY:</span>
                <span className="font-bold text-[#00f5a0]">{record?.total_latency_ms != null ? `${record.total_latency_ms.toFixed(2)} ms` : '--'}</span>
              </div>
            </div>

            {/* Decision Explanation */}
            <div className="p-2.5 rounded bg-[#060a14] border border-[#1e2e48] text-xs font-mono text-[#dae2fd]">
              <div className="text-[10px] text-[#849588] font-bold">RATIONALE:</div>
              <div className="mt-0.5 text-[11px] leading-relaxed">
                {record?.decision_reason || 'No decision telemetry logged.'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
