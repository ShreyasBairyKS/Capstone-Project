import React, { useState, useEffect, useCallback } from 'react';
import { 
  BrainCircuit, 
  Download, 
  RefreshCw, 
  CheckCircle2, 
  AlertTriangle, 
  AlertOctagon, 
  Save, 
  Droplets, 
  Layers,
  ChevronRight,
  Sliders
} from 'lucide-react';

export default function FillActiveLearningPage({ initialBottleId, onSelectBottle }) {
  const [candidates, setCandidates] = useState([]);
  const [selectedId, setSelectedId] = useState(initialBottleId || '');
  const [bottleDetail, setBottleDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [successMsg, setSuccessMsg] = useState('');

  // Operator Adjustment States
  const [verifiedVerdict, setVerifiedVerdict] = useState('NORMAL');
  const [fillStatus, setFillStatus] = useState('normal');
  const [adjustedWaterY, setAdjustedWaterY] = useState(150);
  const [operatorNotes, setOperatorNotes] = useState('');

  // Fetch candidate queue
  const fetchCandidates = useCallback(async () => {
    try {
      const res = await fetch('/api/fill/active-learning/candidates?uncertain_only=false&limit=50');
      if (res.ok) {
        const list = await res.json();
        setCandidates(list || []);
        if (!selectedId && list.length > 0) {
          setSelectedId(list[0].bottle_id);
        }
      }
    } catch (err) {
      console.error('Error fetching AL candidates:', err);
    }
  }, [selectedId]);

  // Fetch selected bottle detail
  const fetchBottle = useCallback(async (id) => {
    if (!id) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/fill/bottle/${encodeURIComponent(id)}`);
      if (res.ok) {
        const json = await res.json();
        setBottleDetail(json);
        const r = json.record;
        if (r) {
          setVerifiedVerdict(r.global_verdict === 'REJECT' ? 'REJECT' : 'NORMAL');
          setFillStatus(r.fill_status || 'normal');
          setAdjustedWaterY(r.water_y ?? Math.round((json.image_size?.height || 500) * 0.5));
        }
      }
    } catch (err) {
      console.error('Error fetching bottle for AL:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCandidates();
  }, [fetchCandidates]);

  useEffect(() => {
    if (selectedId) {
      fetchBottle(selectedId);
    }
  }, [selectedId, fetchBottle]);

  // Handle Save Annotation
  const handleSaveAnnotation = async () => {
    if (!selectedId) return;
    setSaving(true);
    setSuccessMsg('');
    try {
      // Keep the model-detected horizontal extent of the meniscus; only the operator-adjusted Y moves.
      const meta = bottleDetail?.metadata || {};
      const srcBox = meta.meniscus_box || meta.bottle_box;
      const x1 = srcBox ? srcBox[0] : Math.round(imageWidth * 0.25);
      const x2 = srcBox ? srcBox[2] : Math.round(imageWidth * 0.75);
      const payload = {
        bottle_id: selectedId,
        global_verdict: verifiedVerdict === 'NORMAL' ? 'PASS' : 'REJECT',
        fill_status: fillStatus,
        water_box: [x1, Math.max(0, adjustedWaterY - 5), x2, Math.min(imageHeight, adjustedWaterY + 5)]
      };

      const res = await fetch('/api/fill/active-learning/annotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        setSuccessMsg('Annotation committed to YOLO labels and SQLite DB!');
        setTimeout(() => setSuccessMsg(''), 3500);
        fetchCandidates();
      }
    } catch (err) {
      console.error('Error committing AL annotation:', err);
    } finally {
      setSaving(false);
    }
  };

  // Handle Export ZIP
  const handleExportZip = async () => {
    setExporting(true);
    try {
      const res = await fetch('/api/fill/active-learning/export-zip');
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `fill_yolo_dataset_${new Date().toISOString().slice(0, 10)}.zip`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      }
    } catch (err) {
      console.error('Error exporting dataset ZIP:', err);
    } finally {
      setExporting(false);
    }
  };

  const record = bottleDetail?.record;
  const rawImage = bottleDetail?.raw_image_base64;
  // Meniscus Y is stored in original image pixels, so the canvas must scale to the real frame height
  const imageWidth = bottleDetail?.image_size?.width || 640;
  const imageHeight = bottleDetail?.image_size?.height || 500;

  return (
    <div className="flex flex-col gap-3.5 w-full">
      {/* Top Banner */}
      <div className="p-3.5 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-wrap items-center justify-between gap-3 shadow-sm">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded bg-[#17253d] border border-[#fbbf24]/40 text-[#fbbf24]">
            <BrainCircuit size={18} />
          </div>
          <div>
            <div className="text-xs font-mono font-bold text-[#dae2fd] flex items-center gap-2">
              <span>STATION 2 ACTIVE LEARNING &amp; LIQUID MENISCUS CALIBRATION</span>
              <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-[#fbbf24]/10 text-[#fbbf24] border border-[#fbbf24]/30">
                S3 WORKBENCH
              </span>
            </div>
            <div className="text-[10px] font-mono text-[#849588]">
              HUMAN-IN-THE-LOOP TRIAGE FOR BORDERLINE LIQUID LEVEL DETECTIONS
            </div>
          </div>
        </div>

        {/* Right Action: Export YOLO Dataset ZIP */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => fetchCandidates()}
            className="p-1.5 rounded bg-[#17253d] border border-[#1e2e48] text-[#849588] hover:text-[#dae2fd] transition-colors"
            title="Refresh Candidates"
          >
            <RefreshCw size={14} />
          </button>

          <button
            onClick={handleExportZip}
            disabled={exporting}
            className="px-3 py-1.5 rounded bg-[#17253d] border border-[#00daf3]/40 text-[#00daf3] hover:bg-[#1f3354] transition-colors text-xs font-mono font-bold flex items-center gap-1.5"
          >
            <Download size={13} />
            <span>{exporting ? 'PACKING ZIP...' : 'EXPORT YOLOv11 ZIP'}</span>
          </button>
        </div>
      </div>

      {/* Main Grid: Left Candidates Queue, Right Editor */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5">
        {/* Candidates Queue (4 Cols) */}
        <div className="lg:col-span-4 p-3 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-col gap-2 max-h-[640px] shadow-sm">
          <div className="flex items-center justify-between border-b border-[#1e2e48] pb-1.5 text-xs font-mono">
            <span className="font-bold text-[#dae2fd]">TRIAGE QUEUE ({candidates.length})</span>
            <span className="text-[10px] text-[#849588]">SELECT SPECIMEN</span>
          </div>

          <div className="flex flex-col gap-1.5 overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-[#1e2e48]">
            {candidates.length === 0 ? (
              <div className="p-6 text-center text-[#849588] text-xs font-mono">
                No active learning specimens in queue.
              </div>
            ) : (
              candidates.map((c) => {
                const isSelected = c.bottle_id === selectedId;
                const ratio = c.fill_ratio != null ? (c.fill_ratio * 100).toFixed(1) : '--';
                const isPass = c.global_verdict === 'PASS';
                const isReject = c.global_verdict === 'REJECT';

                return (
                  <div
                    key={c.bottle_id}
                    onClick={() => {
                      setSelectedId(c.bottle_id);
                      onSelectBottle?.(c.bottle_id);
                    }}
                    className={`p-2.5 rounded border cursor-pointer transition-all duration-150 flex items-center justify-between
                      ${isSelected 
                        ? 'bg-[#17253d] border-[#00daf3] shadow-[0_0_8px_rgba(0,218,243,0.2)]' 
                        : 'bg-[#070d1c] border-[#1e2e48] hover:bg-[#131b2e]'}`
                    }
                  >
                    <div className="flex flex-col gap-0.5 truncate max-w-[180px]">
                      <span className="text-xs font-mono font-bold text-[#dae2fd] truncate">
                        {c.bottle_id}
                      </span>
                      <span className="text-[10px] font-mono text-[#849588]">
                        {c.timestamp ? c.timestamp.slice(11, 19) : '--'} &bull; {c.fill_status}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold
                        ${isPass ? 'bg-[#00f5a0]/15 text-[#00f5a0]' : ''}
                        ${isReject ? 'bg-[#ff3b30]/15 text-[#ff817b]' : ''}
                        ${c.global_verdict === 'UNCERTAIN' ? 'bg-[#f59e0b]/15 text-[#fbbf24]' : ''}`
                      }>
                        {ratio}%
                      </span>
                      <ChevronRight size={14} className={isSelected ? 'text-[#00daf3]' : 'text-[#849588]'} />
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Annotation & Calibration Editor (8 Cols) */}
        <div className="lg:col-span-8 flex flex-col gap-3">
          <div className="p-3.5 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-col gap-3 shadow-sm">
            <div className="flex items-center justify-between border-b border-[#1e2e48] pb-2 text-xs font-mono">
              <span className="font-bold text-[#dae2fd] flex items-center gap-1.5">
                <Sliders size={14} className="text-[#00daf3]" />
                MENISCUS CALIBRATION CANVAS &bull; {selectedId || 'NO SELECTION'}
              </span>
              <span className="text-[10px] text-[#849588]">OPERATOR GROUND TRUTH</span>
            </div>

            {/* Split Preview / Canvas */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
              {/* Image Preview with Interactive Meniscus Line Guide */}
              <div className="relative rounded bg-[#060a14] border border-[#1e2e48] p-2 flex flex-col items-center justify-center min-h-[360px] overflow-hidden">
                {rawImage ? (
                  <div className="relative max-h-[380px] w-auto">
                    <img 
                      src={rawImage} 
                      alt="Raw Bottle" 
                      className="max-h-[380px] w-auto object-contain rounded"
                    />
                    {/* Visual Calibrated Meniscus Guide */}
                    <div 
                      className="absolute left-4 right-4 h-0.5 bg-[#00daf3] shadow-[0_0_8px_#00daf3] pointer-events-none"
                      style={{ top: `${Math.min(100, Math.max(0, (adjustedWaterY / imageHeight) * 100))}%` }}
                    >
                      <span className="absolute -top-3 right-0 text-[9px] font-mono bg-[#060a14] text-[#00daf3] px-1 rounded border border-[#00daf3]/40">
                        Y: {adjustedWaterY}px
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="text-[#849588] text-xs font-mono">Select a specimen from queue</div>
                )}
              </div>

              {/* Calibration Controls */}
              <div className="p-3 rounded bg-[#070d1c] border border-[#1e2e48] flex flex-col justify-between gap-3">
                <div className="flex flex-col gap-3">
                  <div className="text-[10px] font-mono font-bold text-[#849588] uppercase tracking-wider">
                    OPERATOR VERIFICATION CONTROLS
                  </div>

                  {/* Verdict Select */}
                  <div className="flex flex-col gap-1">
                    <label className="text-xs font-mono text-[#dae2fd]">VERIFIED VERDICT:</label>
                    <div className="grid grid-cols-2 gap-1.5">
                      <button
                        onClick={() => {
                          setVerifiedVerdict('NORMAL');
                          setFillStatus('normal');
                        }}
                        className={`p-2 rounded text-xs font-mono font-bold border transition-colors flex items-center justify-center gap-1.5
                          ${verifiedVerdict === 'NORMAL'
                            ? 'bg-[#00f5a0]/15 text-[#00f5a0] border-[#00f5a0]'
                            : 'bg-[#131b2e] text-[#849588] border-[#1e2e48]'}`
                        }
                      >
                        <CheckCircle2 size={13} />
                        <span>NORMAL (PASS)</span>
                      </button>

                      <button
                        onClick={() => {
                          setVerifiedVerdict('REJECT');
                          setFillStatus('underfill');
                        }}
                        className={`p-2 rounded text-xs font-mono font-bold border transition-colors flex items-center justify-center gap-1.5
                          ${verifiedVerdict === 'REJECT'
                            ? 'bg-[#ff3b30]/15 text-[#ff817b] border-[#ff3b30]'
                            : 'bg-[#131b2e] text-[#849588] border-[#1e2e48]'}`
                        }
                      >
                        <AlertOctagon size={13} />
                        <span>REJECT</span>
                      </button>
                    </div>
                  </div>

                  {/* Fill Classification Dropdown */}
                  <div className="flex flex-col gap-1">
                    <label className="text-xs font-mono text-[#dae2fd]">FILL STATUS CLASSIFICATION:</label>
                    <select
                      value={fillStatus}
                      onChange={e => setFillStatus(e.target.value)}
                      className="p-2 rounded bg-[#060a14] border border-[#1e2e48] text-xs font-mono text-[#dae2fd] focus:outline-none focus:border-[#00daf3]"
                    >
                      <option value="normal">Normal (Nominal Volume 40-85%)</option>
                      <option value="underfill">Underfill (&lt; 40%)</option>
                      <option value="overfill">Overfill (&gt; 85%)</option>
                      <option value="no_water">Dry Container (No Water)</option>
                    </select>
                  </div>

                  {/* Meniscus Y Height Calibration Slider */}
                  <div className="flex flex-col gap-1">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-[#dae2fd]">MENISCUS SURFACE Y:</span>
                      <span className="font-bold text-[#00daf3]">{adjustedWaterY} px</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={imageHeight}
                      value={adjustedWaterY}
                      onChange={e => setAdjustedWaterY(parseInt(e.target.value, 10))}
                      className="w-full accent-[#00daf3] cursor-pointer"
                    />
                    <div className="flex justify-between text-[9px] font-mono text-[#849588]">
                      <span>TOP (OVERFILL)</span>
                      <span>BOTTOM (UNDERFILL)</span>
                    </div>
                  </div>

                  {/* Operator Notes */}
                  <div className="flex flex-col gap-1">
                    <label className="text-xs font-mono text-[#dae2fd]">AUDIT NOTES:</label>
                    <textarea
                      rows={2}
                      placeholder="Optional human-in-the-loop audit notes..."
                      value={operatorNotes}
                      onChange={e => setOperatorNotes(e.target.value)}
                      className="p-2 rounded bg-[#060a14] border border-[#1e2e48] text-xs font-mono text-[#dae2fd] placeholder-[#849588] focus:outline-none focus:border-[#00daf3] resize-none"
                    />
                  </div>
                </div>

                {/* Commit Button */}
                <div className="flex flex-col gap-1.5 pt-2 border-t border-[#1e2e48]">
                  {successMsg && (
                    <div className="text-[11px] font-mono text-[#00f5a0] flex items-center gap-1.5 animate-pulse">
                      <CheckCircle2 size={13} />
                      <span>{successMsg}</span>
                    </div>
                  )}

                  <button
                    onClick={handleSaveAnnotation}
                    disabled={saving || !selectedId}
                    className="w-full py-2.5 rounded bg-[#00daf3]/15 text-[#00daf3] hover:bg-[#00daf3]/25 border border-[#00daf3]/50 disabled:opacity-40 text-xs font-mono font-bold flex items-center justify-center gap-2 transition-all shadow-[0_0_12px_rgba(0,218,243,0.15)]"
                  >
                    <Save size={14} />
                    <span>{saving ? 'COMMITTING TO YOLO...' : 'COMMIT GROUND-TRUTH TO DATASET'}</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
