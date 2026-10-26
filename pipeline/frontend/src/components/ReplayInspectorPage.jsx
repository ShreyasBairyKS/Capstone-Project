import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Eye,
  Layers,
  Camera,
  Download,
  FileText,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Crosshair,
  Sliders,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  ShieldAlert,
  ShieldCheck,
  Cpu,
  Clock,
  ArrowRight,
  Search,
  ExternalLink,
  SplitSquareVertical,
  Maximize,
  Move,
  Play,
  Pause,
  SlidersHorizontal,
  Check,
  Info
} from 'lucide-react';

// Same palette as the backend renderer (visualizer/schemas.py), so a class always has the same color
const CLASS_COLORS = {
  good_cap: '#2ecc71',
  damaged_cap: '#e72e2e',
  misplaced_cap: '#e7782e',
  open_cap: '#f0461e',
  wet_cap: '#34a0eb',
  no_cap: '#c81e1e'
};
const classColor = (name) => CLASS_COLORS[name] || '#ffd84d';

// A detection is shown if it meets the gate's confidence floor or is the one that decided its camera's verdict
function filterViewDetections(viewData, minConf, showLowConf) {
  const dets = viewData?.detections || [];
  if (showLowConf) return dets;
  return dets.filter(d =>
    d.confidence >= minConf ||
    (d.class_name === viewData?.primary_class && Math.abs(d.confidence - (viewData?.primary_confidence ?? -1)) < 1e-6)
  );
}

export default function ReplayInspectorPage({
  initialBottleId,
  onSelectBottle,
  onTriageToAnnotation
}) {
  // Current inspection state
  const [currentBottleId, setCurrentBottleId] = useState(initialBottleId || null);
  const [inspectionData, setInspectionData] = useState(null);
  const [adjacentSequence, setAdjacentSequence] = useState({ previous: [], next: [] });
  const [candidateList, setCandidateList] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Viewport & Layout state
  const [layoutMode, setLayoutMode] = useState('dual'); // 'dual' (split screen) | 'single' (focused)
  const [activeCam, setActiveCam] = useState('camera_front');
  const [renderOverlay, setRenderOverlay] = useState(true);
  const [maskAlpha, setMaskAlpha] = useState(0.5);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showLowConf, setShowLowConf] = useState(false);
  const [minConf, setMinConf] = useState(0.45);

  // Display floor = the quality gate's defect confidence threshold
  useEffect(() => {
    fetch('/api/config')
      .then(r => (r.ok ? r.json() : null))
      .then(cfg => { if (cfg?.defect_conf_threshold) setMinConf(cfg.defect_conf_threshold); })
      .catch(() => null);
  }, []);
  const [showMasks, setShowMasks] = useState(true);
  const [showCrosshairs, setShowCrosshairs] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [syncZoom, setSyncZoom] = useState(true);

  // Zoom & Pan state
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const [cursorNorm, setCursorNorm] = useState({ x: 0, y: 0 });

  // Focused detection contour
  const [focusedDetection, setFocusedDetection] = useState(null);

  // Auto-play slow-mo timeline sequence
  const [isPlayingSequence, setIsPlayingSequence] = useState(false);
  const playIntervalRef = useRef(null);

  // Status notification
  const [notification, setNotification] = useState('');

  // ---------------------------------------------------------------------------
  // 1. Fetch Candidates List & Initial Specimen
  // ---------------------------------------------------------------------------

  const showToast = (msg) => {
    setNotification(msg);
    setTimeout(() => setNotification(prev => prev === msg ? '' : prev), 3500);
  };

  const fetchRecentInspections = useCallback(async () => {
    try {
      const res = await fetch('/api/defects?limit=50&paginate=true');
      if (res.ok) {
        const data = await res.json();
        const items = data.items || [];
        setCandidateList(items);

        // If no bottle ID was selected, pick the latest one
        if (!currentBottleId && items.length > 0) {
          setCurrentBottleId(items[0].bottle_id);
        }
      }
    } catch (err) {
      console.error('Failed to load inspection list:', err);
    }
  }, [currentBottleId]);

  useEffect(() => {
    fetchRecentInspections();
  }, [fetchRecentInspections]);

  // If initialBottleId changes from parent prop, update current bottle
  useEffect(() => {
    if (initialBottleId && initialBottleId !== currentBottleId) {
      setCurrentBottleId(initialBottleId);
    }
  }, [initialBottleId]);

  // ---------------------------------------------------------------------------
  // 2. Load Bottle Detail & Adjacent Sequence
  // ---------------------------------------------------------------------------

  const loadBottleInspection = useCallback(async (bottleId) => {
    if (!bottleId) return;
    setIsLoading(true);
    try {
      const [detailRes, adjRes] = await Promise.all([
        fetch(`/api/bottles/${encodeURIComponent(bottleId)}`),
        fetch(`/api/bottles/${encodeURIComponent(bottleId)}/adjacent?window=4`)
      ]);

      if (detailRes.ok) {
        const detailData = await detailRes.json();
        setInspectionData(detailData);

        // Auto-select first available camera
        const manifestCams = Object.keys(detailData?.manifest?.bundle_manifest?.images || {});
        if (manifestCams.length > 0 && !manifestCams.includes(activeCam)) {
          setActiveCam(manifestCams[0]);
        }
      }

      if (adjRes.ok) {
        const adjData = await adjRes.json();
        setAdjacentSequence({
          previous: adjData.previous || [],
          next: adjData.next || []
        });
      }

      // Reset zoom/pan on specimen switch
      setZoom(1.0);
      setPan({ x: 0, y: 0 });
      setFocusedDetection(null);
    } catch (err) {
      console.error(`Failed to load bottle ${bottleId}:`, err);
      showToast(`Error loading bottle #${bottleId}`);
    } finally {
      setIsLoading(false);
    }
  }, [activeCam]);

  useEffect(() => {
    if (currentBottleId) {
      loadBottleInspection(currentBottleId);
    }
  }, [currentBottleId, loadBottleInspection]);

  // ---------------------------------------------------------------------------
  // 3. Chronological Stepping Controls
  // ---------------------------------------------------------------------------

  const handleStepPrevious = () => {
    if (adjacentSequence.previous && adjacentSequence.previous.length > 0) {
      const prevBottle = adjacentSequence.previous[adjacentSequence.previous.length - 1];
      setCurrentBottleId(prevBottle.bottle_id);
    } else {
      showToast('At the earliest recorded bottle in this sequence');
    }
  };

  const handleStepNext = () => {
    if (adjacentSequence.next && adjacentSequence.next.length > 0) {
      const nextBottle = adjacentSequence.next[0];
      setCurrentBottleId(nextBottle.bottle_id);
    } else {
      showToast('At the latest recorded bottle in this sequence');
    }
  };

  const handleJumpToLatest = async () => {
    try {
      const res = await fetch('/api/bottles/latest');
      if (res.ok) {
        const data = await res.json();
        const bId = data?.record?.bottle_id;
        if (bId) {
          setCurrentBottleId(bId);
          showToast(`Jumped to latest inspection (#${bId.slice(-6)})`);
        }
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Auto-play timeline slow-mo sequence
  const togglePlaySequence = () => {
    if (isPlayingSequence) {
      clearInterval(playIntervalRef.current);
      setIsPlayingSequence(false);
      showToast('Timeline sequence paused');
    } else {
      setIsPlayingSequence(true);
      showToast('Playing chronological sequence (1.5s step)');
      playIntervalRef.current = setInterval(() => {
        setAdjacentSequence(curr => {
          if (curr.next && curr.next.length > 0) {
            setCurrentBottleId(curr.next[0].bottle_id);
          } else {
            clearInterval(playIntervalRef.current);
            setIsPlayingSequence(false);
          }
          return curr;
        });
      }, 1500);
    }
  };

  useEffect(() => {
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, []);

  // ---------------------------------------------------------------------------
  // 4. Keyboard Shortcuts Hook
  // ---------------------------------------------------------------------------

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;

      const key = e.key.toLowerCase();

      // Step Previous: Left Arrow or A
      if (e.key === 'ArrowLeft' || key === 'a') {
        e.preventDefault();
        handleStepPrevious();
      }
      // Step Next: Right Arrow or D
      else if (e.key === 'ArrowRight' || key === 'd') {
        e.preventDefault();
        handleStepNext();
      }
      // Space: Play / Pause sequence
      else if (e.code === 'Space') {
        e.preventDefault();
        togglePlaySequence();
      }
      // T: Triage to S3 Annotation Station
      else if (key === 't') {
        e.preventDefault();
        handleTriage();
      }
      // R: Toggle Raw vs Overlay
      else if (key === 'r') {
        e.preventDefault();
        setRenderOverlay(prev => !prev);
        showToast(`Render mode: ${!renderOverlay ? 'Overlay Mask' : 'Clean Raw'}`);
      }
      // C: Toggle Reticle Crosshair
      else if (key === 'c') {
        e.preventDefault();
        setShowCrosshairs(prev => !prev);
      }
      // Zoom hotkeys
      else if (key === '+' || key === '=') {
        e.preventDefault();
        setZoom(z => Math.min(5.0, z + 0.25));
      } else if (key === '-' || key === '_') {
        e.preventDefault();
        setZoom(z => Math.max(0.8, z - 0.25));
      } else if (key === '0') {
        e.preventDefault();
        setZoom(1.0);
        setPan({ x: 0, y: 0 });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [adjacentSequence, renderOverlay, currentBottleId]);

  // ---------------------------------------------------------------------------
  // 5. Canvas Zoom & Pan Mouse Wheel / Drag
  // ---------------------------------------------------------------------------

  const handleWheel = (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.88;
    setZoom(prev => Math.max(0.8, Math.min(5.0, prev * factor)));
  };

  const handleMouseDownCanvas = (e) => {
    if (e.button === 0 || e.button === 1 || isSpacePressed) {
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMoveCanvas = (e) => {
    if (isPanning) {
      setPan({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y
      });
    }
  };

  const handleMouseUpCanvas = () => {
    setIsPanning(false);
  };

  // ---------------------------------------------------------------------------
  // 6. Direct Triage to Annotation Station (S3)
  // ---------------------------------------------------------------------------

  const handleTriage = () => {
    if (!currentBottleId) return;
    if (onTriageToAnnotation) {
      onTriageToAnnotation(currentBottleId);
    }
  };

  // ---------------------------------------------------------------------------
  // 7. Export Manifest JSON
  // ---------------------------------------------------------------------------

  const handleExportJSON = () => {
    if (!inspectionData) return;
    const blob = new Blob([JSON.stringify(inspectionData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `inspection_forensics_${currentBottleId}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    showToast('Inspection JSON manifest exported');
  };

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  const record = inspectionData?.record || {};
  const manifest = inspectionData?.manifest || {};
  const availableCams = Object.keys(manifest?.bundle_manifest?.images || { camera_front: 'camera_front.jpg' });
  const rawPerViewResults = manifest?.per_view_results || {};
  // Per-camera detections after the low-confidence filter; every consumer below uses this same list
  const perViewResults = Object.fromEntries(
    Object.entries(rawPerViewResults).map(([cam, v]) => [cam, { ...v, detections: filterViewDetections(v, minConf, showLowConf) }])
  );

  const globalVerdict = record.global_verdict || manifest.global_verdict || 'PASS';
  const primaryDefect = record.primary_defect || manifest.primary_defect;
  const confidencePct = record.primary_defect_confidence ? Math.round(record.primary_defect_confidence * 100) : null;
  const actuatorFired = record.trigger_actuator || manifest.trigger_reject_actuator;

  // Flatten detections across cameras for the interactive inspector list (each row is tagged with its camera)
  const allDetections = [];
  Object.entries(perViewResults).forEach(([camId, viewData]) => {
    const dets = viewData.detections || [];
    dets.forEach((det, idx) => {
      allDetections.push({
        ...det,
        camera_id: camId,
        uid: `${camId}_det_${idx}`,
        view_verdict: viewData.verdict
      });
    });
  });

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] overflow-hidden bg-[#060e20] text-[#dae2fd] border border-[#1e2e48] rounded select-none">
      {/* =====================================================================
          TOP TELEMETRY & CHRONOLOGICAL STEPPER HEADER
      ===================================================================== */}
      <header className="h-12 bg-[#0b1326] border-b border-[#1e2e48] px-3 flex items-center justify-between gap-2.5 shrink-0 z-30">
        {/* Left: Specimen Identifier & Search */}
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <div className="flex items-center gap-1 shrink-0">
            <span className="font-['Outfit'] font-bold text-sm text-[#00daf3] tracking-wide uppercase">VisionQ S2</span>
            <span className="hidden xl:inline-block font-mono text-[9px] px-1 py-0.5 bg-[#171f33] border border-[#1e2e48] text-[#849588]">FORENSIC-REPLAY</span>
          </div>

          <div className="h-4 w-px bg-[#1e2e48] shrink-0" />

          {/* Specimen Dropdown */}
          <div className="relative min-w-[130px] max-w-[200px] xl:min-w-[200px] xl:max-w-xs flex-1">
            <select
              value={currentBottleId || ''}
              onChange={(e) => setCurrentBottleId(e.target.value)}
              className="w-full bg-[#060e20] border border-[#1e2e48] px-2 py-1 rounded text-xs font-mono text-[#00daf3] focus:outline-none focus:border-[#00daf3] truncate cursor-pointer"
            >
              {candidateList.map((cand) => (
                <option key={cand.bottle_id} value={cand.bottle_id}>
                  {cand.bottle_id} | {cand.global_verdict} {cand.primary_defect ? `(${cand.primary_defect})` : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Quick Search Input */}
          <div className="relative w-28 xl:w-36 shrink-0">
            <Search className="absolute left-2 top-2 text-[#849588]" size={12} />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && searchQuery.trim()) {
                  setCurrentBottleId(searchQuery.trim());
                  showToast(`Searching for #${searchQuery.trim()}...`);
                }
              }}
              placeholder="Search ID..."
              className="w-full bg-[#060e20] border border-[#1e2e48] pl-6 pr-2 py-1 rounded text-xs font-mono text-[#dae2fd] placeholder-[#849588] focus:outline-none focus:border-[#00daf3]"
            />
          </div>
        </div>

        {/* Center: Chronological Stepper Controls */}
        <div className="flex items-center gap-1 bg-[#060e20] p-0.5 border border-[#1e2e48] rounded shrink-0">
          <button
            onClick={handleStepPrevious}
            className="p-1 rounded bg-[#171f33] border border-[#1e2e48] text-[#849588] hover:text-[#00daf3] transition-colors"
            title="Previous Bottle [← / A]"
          >
            <ChevronLeft size={13} />
          </button>

          <button
            onClick={togglePlaySequence}
            className={`px-2 py-0.5 rounded font-mono text-[10px] font-bold flex items-center gap-1 border transition-colors ${
              isPlayingSequence
                ? 'bg-[#00daf3]/20 text-[#00daf3] border-[#00daf3]/50 animate-pulse'
                : 'bg-[#171f33] text-[#dae2fd] border-[#1e2e48] hover:text-[#00daf3]'
            }`}
            title="Auto-Play Sequence (1.5s step) [Space]"
          >
            {isPlayingSequence ? <Pause size={10} /> : <Play size={10} />}
            <span>{isPlayingSequence ? 'PAUSE' : 'SLOW-MO'}</span>
          </button>

          <button
            onClick={handleStepNext}
            className="p-1 rounded bg-[#171f33] border border-[#1e2e48] text-[#849588] hover:text-[#00daf3] transition-colors"
            title="Next Bottle [→ / D]"
          >
            <ChevronRight size={13} />
          </button>

          <div className="h-3 w-px bg-[#1e2e48] mx-0.5" />

          <button
            onClick={handleJumpToLatest}
            className="px-1.5 py-0.5 rounded bg-[#171f33] text-[#849588] hover:text-[#00f5a0] text-[10px] font-mono transition-colors"
            title="Jump to Latest Inspection"
          >
            Latest
          </button>
        </div>

        {/* Right: Decision Badge & Quick Actions */}
        <div className="flex items-center gap-1.5 shrink-0">
          {/* Verdict Badge */}
          <div className={`px-2 py-1 rounded border font-mono text-xs font-bold flex items-center gap-1.5 max-w-[190px] ${
            globalVerdict === 'REJECT'
              ? 'bg-[#ff3b30]/15 text-[#ff817b] border-[#ff3b30]/50 shadow-[0_0_8px_rgba(255,59,48,0.25)]'
              : globalVerdict === 'UNCERTAIN'
              ? 'bg-[#ffb454]/15 text-[#ffb454] border-[#ffb454]/50'
              : 'bg-[#00f5a0]/15 text-[#00f5a0] border-[#00f5a0]/50'
          }`}>
            {globalVerdict === 'REJECT' ? <XCircle size={14} className="shrink-0" /> : globalVerdict === 'UNCERTAIN' ? <AlertTriangle size={14} className="shrink-0" /> : <CheckCircle2 size={14} className="shrink-0" />}
            <span className="truncate">{globalVerdict}: {primaryDefect ? `${primaryDefect} (${confidencePct}%)` : 'COMPLIANT'}</span>
          </div>

          {/* Actuator Status Beacon */}
          <div className={`px-2 py-1 rounded border text-[10px] font-mono flex items-center gap-1.5 shrink-0 ${
            actuatorFired
              ? 'bg-[#ff3b30]/20 text-[#ff817b] border-[#ff3b30]/40'
              : 'bg-[#171f33] text-[#849588] border-[#1e2e48]'
          }`}>
            <span className={`w-2 h-2 rounded-full ${actuatorFired ? 'bg-[#ff3b30] animate-ping' : 'bg-[#849588]'}`} />
            <span>
              {actuatorFired ? (
                <>
                  <span className="xl:inline hidden">BLOW-OFF FIRED</span>
                  <span className="xl:hidden inline">FIRED</span>
                </>
              ) : (
                <>
                  <span className="xl:inline hidden">ACTUATOR STANDBY</span>
                  <span className="xl:hidden inline">STANDBY</span>
                </>
              )}
            </span>
          </div>

          {/* Triage to S3 Button */}
          <button
            onClick={handleTriage}
            className="flex items-center gap-1 px-2 py-1 rounded bg-[#00f5a0]/10 hover:bg-[#00f5a0]/20 border border-[#00f5a0]/40 text-[#00f5a0] font-mono text-xs font-bold transition-all active:scale-[0.98] shrink-0"
            title="Send specimen directly to S3 Active Learning Workbench for human ground-truth editing [T]"
          >
            <ExternalLink size={13} />
            <span className="xl:inline hidden">Triage to Annotation [T]</span>
            <span className="xl:hidden inline">Triage [T]</span>
          </button>

          {/* Export JSON Button */}
          <button
            onClick={handleExportJSON}
            className="p-1.5 rounded bg-[#171f33] border border-[#1e2e48] text-[#849588] hover:text-[#00daf3] transition-colors shrink-0"
            title="Download Full Inspection JSON Manifest"
          >
            <Download size={14} />
          </button>
        </div>
      </header>

      {/* =====================================================================
          MAIN WORKBENCH: 65% VIEWPORT CANVAS + 35% FORENSIC PANEL
      ===================================================================== */}
      <div className="flex-1 flex min-h-0 overflow-hidden relative">
        {/* LEFT / CENTER: Synchronized Viewport Area (65%) */}
        <main className="flex-1 flex flex-col bg-[#050811] relative overflow-hidden border-r border-[#1e2e48]">
          {/* Top Viewport HUD & Controls Bar */}
          <div className="h-8 bg-[#0b1326]/90 backdrop-blur border-b border-[#1e2e48] px-3 flex items-center justify-between text-xs font-mono z-20">
            {/* Viewport Mode & Layer Controls */}
            <div className="flex items-center gap-2">
              {/* Layout Mode Toggle */}
              <div className="flex bg-[#060e20] p-0.5 border border-[#1e2e48] rounded text-[10px]">
                <button
                  onClick={() => setLayoutMode('dual')}
                  className={`px-2 py-0.5 rounded flex items-center gap-1 ${
                    layoutMode === 'dual' ? 'bg-[#171f33] text-[#00daf3] font-bold' : 'text-[#849588] hover:text-[#dae2fd]'
                  }`}
                  title="Dual Synchronized View (CAM-01 & CAM-02)"
                >
                  <SplitSquareVertical size={11} />
                  <span>Dual Split</span>
                </button>
                <button
                  onClick={() => setLayoutMode('single')}
                  className={`px-2 py-0.5 rounded flex items-center gap-1 ${
                    layoutMode === 'single' ? 'bg-[#171f33] text-[#00daf3] font-bold' : 'text-[#849588] hover:text-[#dae2fd]'
                  }`}
                  title="Single Focused Viewport"
                >
                  <Maximize size={11} />
                  <span>Single Focus</span>
                </button>
              </div>

              {/* Single Mode Camera Tabs */}
              {layoutMode === 'single' && (
                <div className="flex bg-[#060e20] p-0.5 border border-[#1e2e48] rounded text-[10px]">
                  {availableCams.map((cam) => (
                    <button
                      key={cam}
                      onClick={() => setActiveCam(cam)}
                      className={`px-2 py-0.5 rounded ${
                        activeCam === cam ? 'bg-[#171f33] text-[#00f5a0] font-bold' : 'text-[#849588] hover:text-[#dae2fd]'
                      }`}
                    >
                      {cam.replace('camera_', '').toUpperCase()}
                    </button>
                  ))}
                </div>
              )}

              <div className="h-3 w-px bg-[#1e2e48]" />

              {/* Raw vs Overlay Toggle */}
              <button
                onClick={() => setRenderOverlay(prev => !prev)}
                className={`px-2 py-0.5 rounded text-[10px] font-bold border transition-colors flex items-center gap-1 ${
                  renderOverlay
                    ? 'bg-[#00f5a0]/15 text-[#00f5a0] border-[#00f5a0]/40'
                    : 'bg-[#171f33] text-[#dae2fd] border-[#1e2e48]'
                }`}
                title="Toggle On-Demand Replay Segmentation Overlay [R]"
              >
                <Layers size={11} />
                <span>{renderOverlay ? 'Dynamic Overlay [ON]' : 'Clean Raw Sensor'}</span>
              </button>

              {/* Alpha Slider */}
              {renderOverlay && (
                <div className="flex items-center gap-1 text-[10px] text-[#849588]">
                  <span>α:</span>
                  <input
                    type="range"
                    min="0.2"
                    max="0.8"
                    step="0.05"
                    value={maskAlpha}
                    onChange={(e) => setMaskAlpha(parseFloat(e.target.value))}
                    className="w-14 h-1 accent-[#00f5a0] cursor-pointer"
                    title="Adjust segmentation mask transparency alpha"
                  />
                  <span>{(maskAlpha * 100).toFixed(0)}%</span>
                </div>
              )}

              {/* Layer Toggles */}
              <button
                onClick={() => setShowLowConf(prev => !prev)}
                className={`px-1.5 py-0.5 rounded text-[10px] border ${showLowConf ? 'bg-[#171f33] text-[#fbbf24] border-[#fbbf24]/40' : 'text-[#849588] border-[#1e2e48]'}`}
                title={`Show detections below the ${(minConf * 100).toFixed(0)}% gate confidence floor`}
              >
                Low-conf [{showLowConf ? 'ON' : 'OFF'}]
              </button>

              <button
                onClick={() => setShowBoxes(prev => !prev)}
                className={`px-1.5 py-0.5 rounded text-[10px] border ${showBoxes ? 'bg-[#171f33] text-[#00daf3] border-[#00daf3]/40' : 'text-[#849588] border-[#1e2e48]'}`}
                title="Toggle Bounding Boxes"
              >
                Box [{showBoxes ? 'ON' : 'OFF'}]
              </button>

              <button
                onClick={() => setShowCrosshairs(prev => !prev)}
                className={`px-1.5 py-0.5 rounded text-[10px] border ${showCrosshairs ? 'bg-[#171f33] text-[#00daf3] border-[#00daf3]/40' : 'text-[#849588] border-[#1e2e48]'}`}
                title="Toggle Reticle Crosshairs [C]"
              >
                <Crosshair size={10} className="inline mr-0.5" />
                Crosshair
              </button>
            </div>

            {/* Zoom & Pan Controls */}
            <div className="flex items-center gap-1 text-[10px]">
              <span className="text-[#849588]">ZOOM:</span>
              <button
                onClick={() => setZoom(z => Math.max(0.8, z - 0.25))}
                className="w-5 h-5 flex items-center justify-center bg-[#171f33] border border-[#1e2e48] text-[#849588] hover:text-[#00daf3] rounded"
              >
                -
              </button>
              <span className="text-[#00f5a0] font-bold px-1">{(zoom * 100).toFixed(0)}%</span>
              <button
                onClick={() => setZoom(z => Math.min(5.0, z + 0.25))}
                className="w-5 h-5 flex items-center justify-center bg-[#171f33] border border-[#1e2e48] text-[#849588] hover:text-[#00daf3] rounded"
              >
                +
              </button>
              <button
                onClick={() => { setZoom(1.0); setPan({ x: 0, y: 0 }); }}
                className="px-1.5 py-0.5 bg-[#171f33] border border-[#1e2e48] text-[#849588] hover:text-[#dae2fd] rounded ml-1"
                title="Reset 1:1 [0]"
              >
                1:1
              </button>
            </div>
          </div>

          {/* Interactive Inspection Viewport Canvas */}
          <div
            onWheel={handleWheel}
            onMouseDown={handleMouseDownCanvas}
            onMouseMove={handleMouseMoveCanvas}
            onMouseUp={handleMouseUpCanvas}
            className={`flex-1 relative overflow-hidden flex items-center justify-center p-3 ${
              isPanning ? 'cursor-grabbing' : 'cursor-grab'
            }`}
            style={{
              backgroundImage: 'radial-gradient(circle, #1a2538 1px, transparent 1px)',
              backgroundSize: '24px 24px'
            }}
          >
            {/* Axis Reticle Crosshair */}
            {showCrosshairs && (
              <>
                <div className="absolute inset-x-0 top-1/2 h-px border-t border-dashed border-[#1e2e48]/60 pointer-events-none z-10" />
                <div className="absolute inset-y-0 left-1/2 w-px border-l border-dashed border-[#1e2e48]/60 pointer-events-none z-10" />
              </>
            )}

            {/* Specimen Frames Viewport Grid */}
            <div
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: 'center center',
                transition: isPanning ? 'none' : 'transform 0.05s ease-out'
              }}
              className="flex items-center justify-center gap-4 max-w-full max-h-full"
            >
              {layoutMode === 'dual' ? (
                // Dual Synchronized Split View
                availableCams.slice(0, 2).map((camKey) => (
                  <CameraViewportPane
                    key={camKey}
                    bottleId={currentBottleId}
                    cameraId={camKey}
                    renderOverlay={renderOverlay}
                    maskAlpha={maskAlpha}
                    showBoxes={showBoxes}
                    showMasks={showMasks}
                    showLabels={showLabels}
                    focusedDetection={focusedDetection}
                    detections={perViewResults[camKey]?.detections || []}
                    onSelectDetection={setFocusedDetection}
                  />
                ))
              ) : (
                // Single Focused Camera View
                <CameraViewportPane
                  bottleId={currentBottleId}
                  cameraId={activeCam}
                  renderOverlay={renderOverlay}
                  maskAlpha={maskAlpha}
                  showBoxes={showBoxes}
                  showMasks={showMasks}
                  showLabels={showLabels}
                  focusedDetection={focusedDetection}
                  detections={perViewResults[activeCam]?.detections || []}
                  onSelectDetection={setFocusedDetection}
                />
              )}
            </div>

            {/* Top-Left Optical Telemetry HUD */}
            <div className="absolute top-3 left-3 bg-[#0b1326]/85 backdrop-blur border border-[#1e2e48] p-2 rounded text-[10px] font-mono pointer-events-none z-20 flex flex-col gap-0.5">
              <div className="text-[#849588] tracking-wider font-bold">OPTICAL SENSOR TELEMETRY</div>
              <div className="text-[#dae2fd]">
                Specimen: <span className="text-[#00daf3] font-bold">#{currentBottleId || '---'}</span>
              </div>
              <div className="text-[#dae2fd]">
                FOV: <span className="text-[#00f5a0]">38.40mm x 38.40mm</span> | Scale: <span className="text-[#00f5a0]">0.0124 mm/px</span>
              </div>
              <div className="text-[#849588]">
                Mode: {layoutMode.toUpperCase()} | Render: {renderOverlay ? 'DYNAMIC OVERLAY' : 'RAW'}
              </div>
            </div>

            {/* In-Canvas Notification Pill */}
            {notification && (
              <div className="absolute top-3 right-3 px-3 py-1.5 rounded font-mono text-xs bg-[#171f33] text-[#00daf3] border border-[#00daf3]/50 shadow-lg z-30 transition-all">
                {notification}
              </div>
            )}
          </div>
        </main>

        {/* RIGHT: Deep-Dive Forensic & Traceability Panel (35%) */}
        <aside className="w-96 bg-[#0b1326] border-l border-[#1e2e48] flex flex-col justify-between overflow-y-auto shrink-0 z-20">
          <div className="p-3.5 flex flex-col gap-3.5">
            {/* SECTION A: Quality Gate Decision Traceability */}
            <div className="bg-[#060e20] border border-[#1e2e48] p-3 rounded">
              <div className="flex items-center justify-between pb-1 mb-2 border-b border-[#1e2e48]">
                <span className="text-[10px] font-mono text-[#849588] uppercase tracking-wider font-bold">
                  QUALITY GATE DECISION TRACEABILITY
                </span>
                <ShieldCheck size={13} className="text-[#00f5a0]" />
              </div>

              <div className="flex flex-col gap-2 font-mono text-xs">
                {/* Global Verdict Row */}
                <div className="flex items-center justify-between">
                  <span className="text-[#849588]">Authoritative Verdict:</span>
                  <span className={`font-bold px-1.5 py-0.5 rounded ${
                    globalVerdict === 'REJECT'
                      ? 'bg-[#ff3b30]/20 text-[#ff817b] border border-[#ff3b30]/40'
                      : globalVerdict === 'UNCERTAIN'
                      ? 'bg-[#ffb454]/20 text-[#ffb454] border border-[#ffb454]/40'
                      : 'bg-[#00f5a0]/20 text-[#00f5a0] border border-[#00f5a0]/40'
                  }`}>
                    {globalVerdict}
                  </span>
                </div>

                {/* Primary Defect & Confidence */}
                <div className="flex items-center justify-between">
                  <span className="text-[#849588]">Primary Defect:</span>
                  <span className="text-[#dae2fd] font-bold">
                    {primaryDefect ? `${primaryDefect} (${confidencePct}%)` : 'NONE (Nominal)'}
                  </span>
                </div>

                {/* Decision Reason */}
                <div className="bg-[#0b1326] p-2 rounded border border-[#1e2e48] text-[11px] text-[#849588] leading-relaxed">
                  <strong className="text-[#dae2fd]">Gate Reason:</strong>{' '}
                  {record.decision_reason || manifest.decision_reason || 'Nominal inspection cycle'}
                </div>

                {/* Latency & Hardware Metrics */}
                <div className="grid grid-cols-2 gap-1.5 text-[10px]">
                  <div className="bg-[#0b1326] p-1.5 rounded border border-[#1e2e48]">
                    <div className="text-[#849588]">CYCLE LATENCY</div>
                    <div className="text-sm font-bold text-[#00daf3] mt-0.5">
                      {record.total_latency_ms ? `${record.total_latency_ms.toFixed(1)}ms` : '---'}
                    </div>
                  </div>
                  <div className="bg-[#0b1326] p-1.5 rounded border border-[#1e2e48]">
                    <div className="text-[#849588]">ACTUATOR EJECT</div>
                    <div className="text-sm font-bold text-[#ff817b] mt-0.5">
                      {actuatorFired ? 'CONFIRMED' : 'BYPASSED'}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* SECTION B: Interactive Detection Contours List */}
            <div>
              <div className="flex items-center justify-between pb-1 border-b border-[#1e2e48] mb-2">
                <span className="text-[10px] font-mono text-[#849588] uppercase tracking-wider font-bold">
                  DETECTED INSTANCE CONTOURS ({allDetections.length})
                </span>
                <span className="text-[9px] font-mono text-[#00daf3]">CLICK TO FOCUS</span>
              </div>

              {allDetections.length === 0 ? (
                <div className="bg-[#060e20] border border-[#1e2e48] p-4 rounded text-center font-mono text-xs text-[#849588]">
                  No detections at or above the confidence floor on this bottle.
                </div>
              ) : (
                <div className="flex flex-col gap-1.5 max-h-56 overflow-y-auto">
                  {allDetections.map((det) => {
                    const isFocused = focusedDetection?.uid === det.uid;
                    const confVal = Math.round(det.confidence * 100);
                    return (
                      <div
                        key={det.uid}
                        onClick={() => setFocusedDetection(det)}
                        className={`p-2 rounded border cursor-pointer font-mono text-xs transition-all ${
                          isFocused
                            ? 'bg-[#171f33] border-[#00daf3] shadow-[0_0_10px_rgba(0,218,243,0.2)]'
                            : 'bg-[#060e20] border-[#1e2e48] hover:border-[#849588]'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-bold text-[#dae2fd] flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-none" style={{ backgroundColor: classColor(det.class_name) }} />
                            <span>{det.class_name}</span>
                          </span>
                          <span className="text-[10px] text-[#00f5a0] font-bold">
                            {confVal}%
                          </span>
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-[10px] text-[#849588]">
                          <div>
                            CAM: <strong className="text-[#dae2fd]">{det.camera_id.replace('camera_', '').toUpperCase()}</strong>
                          </div>
                          <div>
                            NODES: <strong className="text-[#dae2fd]">{det.mask ? det.mask.length : 4} pts</strong>
                          </div>
                        </div>

                        {det.mask_area_pixels && (
                          <div className="text-[10px] text-[#849588] mt-1">
                            Area: <strong className="text-[#00daf3]">{Math.round(det.mask_area_pixels)} px²</strong> (~{(det.mask_area_pixels * 0.00015).toFixed(1)} mm²)
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* SECTION C: Adjacent Production Sequence Filmstrip */}
            <div>
              <div className="flex items-center justify-between pb-1 border-b border-[#1e2e48] mb-2">
                <span className="text-[10px] font-mono text-[#849588] uppercase tracking-wider font-bold">
                  CHRONOLOGICAL TIMELINE SEQUENCE (±4)
                </span>
                <span className="text-[9px] font-mono text-[#849588]">STEP [← / →]</span>
              </div>

              <div className="flex flex-col gap-1 font-mono text-xs">
                {/* Previous Bottles */}
                {adjacentSequence.previous.map((b) => (
                  <div
                    key={b.bottle_id}
                    onClick={() => setCurrentBottleId(b.bottle_id)}
                    className="p-1.5 rounded bg-[#060e20] border border-[#1e2e48] hover:border-[#00daf3] flex items-center justify-between cursor-pointer transition-colors"
                  >
                    <span className="text-[#849588] truncate">{b.bottle_id}</span>
                    <span className={`text-[9px] font-bold px-1 rounded ${
                      b.global_verdict === 'REJECT' ? 'text-[#ff817b] bg-[#ff3b30]/10' : 'text-[#00f5a0] bg-[#00f5a0]/10'
                    }`}>
                      {b.global_verdict}
                    </span>
                  </div>
                ))}

                {/* ACTIVE BOTTLE */}
                <div className="p-2 rounded bg-[#171f33] border-2 border-[#00daf3] flex items-center justify-between shadow-[0_0_10px_rgba(0,218,243,0.15)] font-bold">
                  <span className="text-[#00daf3] truncate">👉 {currentBottleId}</span>
                  <span className={`text-[10px] px-1.5 py-0.2 rounded ${
                    globalVerdict === 'REJECT' ? 'text-[#ff817b] bg-[#ff3b30]/20' : 'text-[#00f5a0] bg-[#00f5a0]/20'
                  }`}>
                    ACTIVE
                  </span>
                </div>

                {/* Next Bottles */}
                {adjacentSequence.next.map((b) => (
                  <div
                    key={b.bottle_id}
                    onClick={() => setCurrentBottleId(b.bottle_id)}
                    className="p-1.5 rounded bg-[#060e20] border border-[#1e2e48] hover:border-[#00daf3] flex items-center justify-between cursor-pointer transition-colors"
                  >
                    <span className="text-[#849588] truncate">{b.bottle_id}</span>
                    <span className={`text-[9px] font-bold px-1 rounded ${
                      b.global_verdict === 'REJECT' ? 'text-[#ff817b] bg-[#ff3b30]/10' : 'text-[#00f5a0] bg-[#00f5a0]/10'
                    }`}>
                      {b.global_verdict}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>
      </div>

      {/* =====================================================================
          BOTTOM STATUS & HOTKEY LEGEND DOCK
      ===================================================================== */}
      <footer className="h-7 bg-[#0b1326] border-t border-[#1e2e48] px-3 flex items-center justify-between text-[11px] font-mono shrink-0 z-30">
        <div className="flex items-center gap-2 overflow-x-auto text-[#849588] whitespace-nowrap">
          <span className="text-[#dae2fd] font-bold mr-1">HOTKEYS:</span>
          <span className="inline-flex items-center gap-1">
            <strong className="bg-[#171f33] border border-[#1e2e48] px-1 py-0.2 rounded text-[#dae2fd]">← / →</strong> Step
          </span>
          <span>·</span>
          <span className="inline-flex items-center gap-1">
            <strong className="bg-[#171f33] border border-[#1e2e48] px-1 py-0.2 rounded text-[#dae2fd]">Space</strong> Slow-mo
          </span>
          <span>·</span>
          <span className="inline-flex items-center gap-1">
            <strong className="bg-[#171f33] border border-[#00f5a0]/40 px-1 py-0.2 rounded text-[#00f5a0]">T</strong> Triage
          </span>
          <span>·</span>
          <span className="inline-flex items-center gap-1">
            <strong className="bg-[#171f33] border border-[#1e2e48] px-1 py-0.2 rounded text-[#dae2fd]">R</strong> Overlay
          </span>
          <span className="hidden xl:inline">·</span>
          <span className="hidden xl:inline-flex items-center gap-1">
            <strong className="bg-[#171f33] border border-[#1e2e48] px-1 py-0.2 rounded text-[#dae2fd]">C</strong> Crosshairs
          </span>
          <span className="hidden 2xl:inline">·</span>
          <span className="hidden 2xl:inline-flex items-center gap-1">
            <strong className="bg-[#171f33] border border-[#1e2e48] px-1 py-0.2 rounded text-[#dae2fd]">0</strong> Reset
          </span>
        </div>

        <div className="flex items-center gap-3 shrink-0 text-[#849588] whitespace-nowrap">
          <span className="hidden lg:inline">GPIO-SYNC: <strong className="text-[#00f5a0]">LOCK 120µs</strong></span>
          <div className="hidden lg:block h-3 w-px bg-[#1e2e48]" />
          <span>INSPECTION ACTIVE</span>
        </div>
      </footer>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Subcomponent: Camera Viewport Pane (Image + Interactive Contours)
// -----------------------------------------------------------------------------
function CameraViewportPane({
  bottleId,
  cameraId,
  renderOverlay,
  maskAlpha,
  showBoxes,
  showMasks,
  showLabels,
  focusedDetection,
  detections = [],
  onSelectDetection
}) {
  const [imgDimensions, setImgDimensions] = useState({ w: 2608, h: 4624 });
  const [imgLoading, setImgLoading] = useState(true);

  useEffect(() => {
    setImgLoading(true);
  }, [bottleId, cameraId, renderOverlay]);

  if (!bottleId) {
    return (
      <div className="w-[360px] h-[480px] bg-[#0b1326] border border-[#1e2e48] flex items-center justify-center font-mono text-xs text-[#849588] rounded">
        No specimen selected
      </div>
    );
  }

  // Always fetch the clean frame: the overlay below is drawn once, per camera, in this component
  // (requesting the server-rendered image as well would paint every box twice).
  const imageUrl = `/api/bottles/${bottleId}/image/${cameraId}?render=false`;
  const defectCount = detections.filter(d => d.class_name !== 'good_cap').length;

  return (
    <div className="relative flex flex-col bg-[#0b1326] border border-[#1e2e48] shadow-2xl rounded overflow-hidden min-w-[320px] max-w-[36vw]">
      {/* Pane Camera Header Tag */}
      <div className="h-6 bg-[#171f33] border-b border-[#1e2e48] px-2.5 flex items-center justify-between gap-2 text-[10px] font-mono shrink-0">
        <span className="text-[#00daf3] font-bold">
          CAM: {cameraId.replace('camera_', '').toUpperCase()}
        </span>
        <span className="text-[#849588]">
          {defectCount} defect(s) &bull; {detections.length} det.
        </span>
      </div>

      {/* Frame Container */}
      <div className="relative min-w-[320px] min-h-[420px] max-h-[68vh] flex items-center justify-center overflow-hidden bg-[#060e20]">
        {imgLoading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-[#060e20] text-[#849588] text-xs font-mono z-10">
            <div className="w-6 h-6 border-2 border-[#00daf3] border-t-transparent rounded-full animate-spin" />
            <span>Loading frame...</span>
          </div>
        )}

        <div className="relative inline-block">
        <img
          src={imageUrl}
          alt={`Specimen ${bottleId} ${cameraId}`}
          onLoad={(e) => {
            setImgLoading(false);
            if (e.target.naturalWidth && e.target.naturalHeight) {
              setImgDimensions({ w: e.target.naturalWidth, h: e.target.naturalHeight });
            }
          }}
          onError={() => setImgLoading(false)}
          className={`max-h-[68vh] max-w-[36vw] object-contain block pointer-events-none select-none transition-opacity duration-200 ${
            imgLoading ? 'opacity-0' : 'opacity-100'
          }`}
        />

        {/* Interactive SVG Contour Layer: sized to the image itself so boxes line up with it */}
        {renderOverlay && (
        <svg
          viewBox={`0 0 ${imgDimensions.w} ${imgDimensions.h}`}
          className="absolute inset-0 w-full h-full pointer-events-auto"
        >
          {detections.map((det, idx) => {
            const isFocused = focusedDetection?.uid === `${cameraId}_det_${idx}`;
            const color = classColor(det.class_name);
            const box = det.box || [];
            const mask = det.mask || [];

            return (
              <g
                key={idx}
                className="cursor-pointer"
                onClick={() => onSelectDetection({ ...det, camera_id: cameraId, uid: `${cameraId}_det_${idx}` })}
              >
                {/* Polygon Mask Outline */}
                {showMasks && mask.length >= 3 && (
                  <polygon
                    points={mask.map(p => `${p[0]},${p[1]}`).join(' ')}
                    fill={isFocused ? '#00daf3' : color}
                    fillOpacity={isFocused ? 0.40 : maskAlpha * 0.5}
                    stroke={isFocused ? '#00daf3' : color}
                    strokeWidth={isFocused ? 6 : 3}
                    className="transition-all hover:fill-opacity-50"
                  />
                )}

                {/* Bounding Box */}
                {showBoxes && box.length === 4 && (
                  <rect
                    x={box[0]}
                    y={box[1]}
                    width={box[2] - box[0]}
                    height={box[3] - box[1]}
                    fill="none"
                    stroke={isFocused ? '#00daf3' : color}
                    strokeWidth={isFocused ? 5 : 2.5}
                    strokeDasharray={isFocused ? '8,4' : 'none'}
                  />
                )}

                {/* Defect Label Tag */}
                {showLabels && box.length === 4 && (
                  <g transform={`translate(${box[0]}, ${Math.max(30, box[1] - 10)})`}>
                    <rect
                      x="0"
                      y="-22"
                      width="260"
                      height="24"
                      fill="#060e20"
                      stroke={isFocused ? '#00daf3' : color}
                      strokeWidth="2"
                    />
                    <text
                      x="8"
                      y="-6"
                      fill="#ffffff"
                      fontFamily="JetBrains Mono, monospace"
                      fontSize="14"
                      fontWeight="bold"
                    >
                      {det.class_name.toUpperCase()} ({(det.confidence * 100).toFixed(0)}%)
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>
        )}
        </div>
      </div>
    </div>
  );
}
