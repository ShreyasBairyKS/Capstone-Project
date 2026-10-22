import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Lock,
  Unlock,
  ZoomIn,
  ZoomOut,
  Maximize2,
  RotateCcw,
  Hand,
  Crop,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Save,
  Download,
  Search,
  Check,
  X,
  ChevronDown,
  RefreshCw,
  Sliders,
  Layers,
  Trash2,
  Crosshair,
  Move,
  Camera,
  ChevronLeft,
  ChevronRight,
  ShieldAlert,
  Info
} from 'lucide-react';

const DEFECT_CLASSES = [
  { id: 0, key: '1', name: 'Damaged Cap', color: '#FF5370', border: 'border-[#FF5370]', bg: 'bg-[#FF5370]' },
  { id: 1, key: '2', name: 'Good Cap', color: '#00F5A0', border: 'border-[#00F5A0]', bg: 'bg-[#00F5A0]' },
  { id: 2, key: '3', name: 'Misplaced Cap', color: '#FFB454', border: 'border-[#FFB454]', bg: 'bg-[#FFB454]' },
  { id: 3, key: '4', name: 'No Cap', color: '#C792EA', border: 'border-[#C792EA]', bg: 'bg-[#C792EA]' },
  { id: 4, key: '5', name: 'Open Cap', color: '#89DDFF', border: 'border-[#89DDFF]', bg: 'bg-[#89DDFF]' },
  { id: 5, key: '6', name: 'Wet Cap', color: '#82AAFF', border: 'border-[#82AAFF]', bg: 'bg-[#82AAFF]' },
];

export default function ActiveLearningPage({ onSelectBottle, initialBottleId }) {
  // Candidate Queue State
  const [candidates, setCandidates] = useState([]);
  const [uncertainOnly, setUncertainOnly] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [uncertainCount, setUncertainCount] = useState(0);
  const [totalCandidateCount, setTotalCandidateCount] = useState(0);
  const [isLoadingCandidates, setIsLoadingCandidates] = useState(false);

  // Active Bottle & Single-Image Lock State
  const [lockedBottleId, setLockedBottleId] = useState(null);
  const [bottleDetail, setBottleDetail] = useState(null);
  const [activeCameraId, setActiveCameraId] = useState(null);
  const [isLoadingBottle, setIsLoadingBottle] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [pendingSwitchBottleId, setPendingSwitchBottleId] = useState(null);
  const [showLockGuardModal, setShowLockGuardModal] = useState(false);

  // Annotation Editor Tool State
  const [activeTool, setActiveTool] = useState('polygon'); // 'polygon', 'box', 'move', 'pan'
  const [selectedClassId, setSelectedClassId] = useState(0); // 0-5
  const [annotations, setAnnotations] = useState([]); // Array of { id, class_id, class_name, points: [[x,y],...] }
  const [draftPoints, setDraftPoints] = useState([]); // Points being drafted in polygon mode
  const [boxStartPoint, setBoxStartPoint] = useState(null); // Box drafting start
  const [boxCurrentPoint, setBoxCurrentPoint] = useState(null); // Box drafting current
  const [activeAnnotationIdx, setActiveAnnotationIdx] = useState(null);
  const [activeVertexIdx, setActiveVertexIdx] = useState(null);
  const [isDraggingVertex, setIsDraggingVertex] = useState(false);

  // Canvas Viewport State (Zoom & Pan)
  const [zoom, setZoom] = useState(1.0); // 0.8 to 5.0
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const [showCrosshairs, setShowCrosshairs] = useState(true);
  const [cursorNormCoord, setCursorNormCoord] = useState({ x: 0, y: 0 });

  // Help & Notification State
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [statusMessage, setStatusMessage] = useState({ text: '', type: 'info' });
  const [isSaving, setIsSaving] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const canvasRef = useRef(null);
  const svgRef = useRef(null);

  // ---------------------------------------------------------------------------
  // 1. Data Fetching
  // ---------------------------------------------------------------------------

  const fetchCandidates = useCallback(async (isUncertain = uncertainOnly, search = searchQuery) => {
    setIsLoadingCandidates(true);
    try {
      const params = new URLSearchParams();
      params.append('uncertain_only', isUncertain ? 'true' : 'false');
      if (search.trim()) params.append('search', search.trim());
      params.append('limit', '60');

      const res = await fetch(`/api/active-learning/candidates?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setCandidates(data.items || []);
        setUncertainCount(data.uncertain_count || 0);
        setTotalCandidateCount(data.total_count || 0);

        // Auto-select first candidate if no bottle is locked yet
        if (!lockedBottleId && data.items && data.items.length > 0) {
          loadBottleForEditing(data.items[0].bottle_id);
        }
      }
    } catch (err) {
      console.error('Failed to fetch candidate bottles:', err);
    } finally {
      setIsLoadingCandidates(false);
    }
  }, [uncertainOnly, searchQuery, lockedBottleId]);

  useEffect(() => {
    fetchCandidates(uncertainOnly, searchQuery);
  }, [uncertainOnly, searchQuery]);

  // If initialBottleId is provided from S2 triage or parent, load it into editor
  useEffect(() => {
    if (initialBottleId && initialBottleId !== lockedBottleId) {
      loadBottleForEditing(initialBottleId);
    }
  }, [initialBottleId]);

  const loadBottleForEditing = async (bottleId) => {
    setIsLoadingBottle(true);
    try {
      const res = await fetch(`/api/active-learning/bottle/${encodeURIComponent(bottleId)}`);
      if (res.ok) {
        const data = await res.json();
        setBottleDetail(data);
        setLockedBottleId(bottleId);
        
        // Setup initial camera
        const camId = data.default_camera || Object.keys(data.cameras || {})[0];
        setActiveCameraId(camId);

        // Load annotations for that camera
        if (data.cameras && data.cameras[camId]) {
          const camAnns = data.cameras[camId].annotations || [];
          setAnnotations(camAnns);
          if (camAnns.length > 0) {
            setSelectedClassId(camAnns[0].class_id);
            setActiveAnnotationIdx(0);
          }
        } else {
          setAnnotations([]);
        }

        // Reset drafting and viewport
        setDraftPoints([]);
        setBoxStartPoint(null);
        setBoxCurrentPoint(null);
        setActiveVertexIdx(null);
        setHasUnsavedChanges(false);
        setZoom(1.0);
        setPan({ x: 0, y: 0 });
        showNotification(`Bottle #${bottleId} locked for editing.`, 'info');
      } else {
        showNotification(`Could not load bottle ${bottleId}.`, 'error');
      }
    } catch (err) {
      console.error('Failed to load bottle detail:', err);
      showNotification('Network error while loading bottle.', 'error');
    } finally {
      setIsLoadingBottle(false);
    }
  };

  // ---------------------------------------------------------------------------
  // 2. Single-Image Lock Enforcement
  // ---------------------------------------------------------------------------

  const handleRequestSwitchBottle = (targetBottleId) => {
    if (targetBottleId === lockedBottleId) return;

    if (hasUnsavedChanges) {
      // Prevent switching until committed or discarded
      setPendingSwitchBottleId(targetBottleId);
      setShowLockGuardModal(true);
    } else {
      // Safe to switch
      loadBottleForEditing(targetBottleId);
    }
  };

  const handleConfirmDiscardAndSwitch = () => {
    setShowLockGuardModal(false);
    if (pendingSwitchBottleId) {
      loadBottleForEditing(pendingSwitchBottleId);
      setPendingSwitchBottleId(null);
    }
  };

  const handleDiscardCurrentLock = () => {
    if (!lockedBottleId) return;
    if (bottleDetail && activeCameraId && bottleDetail.cameras[activeCameraId]) {
      // Revert to original annotations from server
      setAnnotations(bottleDetail.cameras[activeCameraId].annotations || []);
    }
    setDraftPoints([]);
    setBoxStartPoint(null);
    setBoxCurrentPoint(null);
    setActiveVertexIdx(null);
    setHasUnsavedChanges(false);
    showNotification(`Changes discarded. Lock reset for #${lockedBottleId}.`, 'info');
  };

  // ---------------------------------------------------------------------------
  // 3. Switch Camera Angles
  // ---------------------------------------------------------------------------

  const handleSwitchCamera = (camId) => {
    if (camId === activeCameraId) return;
    if (hasUnsavedChanges) {
      const confirmSwitch = window.confirm(
        'You have uncommitted polygon changes on this view. Switching camera views will preserve edits in memory. Continue?'
      );
      if (!confirmSwitch) return;
    }

    setActiveCameraId(camId);
    if (bottleDetail?.cameras?.[camId]) {
      setAnnotations(bottleDetail.cameras[camId].annotations || []);
    } else {
      setAnnotations([]);
    }
    setDraftPoints([]);
    setBoxStartPoint(null);
    setBoxCurrentPoint(null);
    setActiveVertexIdx(null);
  };

  // ---------------------------------------------------------------------------
  // 4. Keyboard Shortcuts Hook
  // ---------------------------------------------------------------------------

  useEffect(() => {
    const handleKeyDown = (e) => {
      // Avoid hotkeys when typing in input/textarea
      if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;

      const key = e.key.toLowerCase();

      // Tool selection shortcuts
      if (key === 'p') {
        setActiveTool('polygon');
        showNotification('Tool: Polygon Segmentation (P)', 'info');
      } else if (key === 'b') {
        setActiveTool('box');
        showNotification('Tool: Bounding Box (B)', 'info');
      } else if (key === 'v') {
        setActiveTool('move');
        showNotification('Tool: Select / Move Node (V)', 'info');
      } else if (key === 'h') {
        setActiveTool('pan');
        showNotification('Tool: Hand / Pan View (H)', 'info');
      }

      // Spacebar temporary pan
      if (e.code === 'Space' && !isSpacePressed) {
        setIsSpacePressed(true);
      }

      // Class palette 1-6
      if (['1', '2', '3', '4', '5', '6'].includes(key)) {
        const classIdx = parseInt(key, 10) - 1;
        if (classIdx >= 0 && classIdx < DEFECT_CLASSES.length) {
          setSelectedClassId(classIdx);
          // If an annotation is actively selected, update its class immediately!
          if (activeAnnotationIdx !== null && annotations[activeAnnotationIdx]) {
            setAnnotations(prev => {
              const next = [...prev];
              next[activeAnnotationIdx] = {
                ...next[activeAnnotationIdx],
                class_id: classIdx,
                class_name: DEFECT_CLASSES[classIdx].name
              };
              return next;
            });
            setHasUnsavedChanges(true);
          }
          showNotification(`Class: ${DEFECT_CLASSES[classIdx].name} [${key}]`, 'info');
        }
      }

      // Ctrl+S or Cmd+S to commit & save
      if ((e.ctrlKey || e.metaKey) && key === 's') {
        e.preventDefault();
        handleCommitAndSave();
      }

      // Escape to cancel draft or deselect
      if (key === 'escape') {
        setDraftPoints([]);
        setBoxStartPoint(null);
        setBoxCurrentPoint(null);
        setActiveVertexIdx(null);
      }

      // Delete or Backspace to delete active vertex or active polygon
      if (key === 'delete' || key === 'backspace') {
        if (activeAnnotationIdx !== null) {
          e.preventDefault();
          if (activeVertexIdx !== null) {
            handleDeleteVertex(activeAnnotationIdx, activeVertexIdx);
          } else {
            handleDeleteAnnotation(activeAnnotationIdx);
          }
        }
      }

      // Help Modal toggle
      if (key === '?' || (e.shiftKey && key === '/')) {
        setShowHelpModal(prev => !prev);
      }
    };

    const handleKeyUp = (e) => {
      if (e.code === 'Space') {
        setIsSpacePressed(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [
    isSpacePressed,
    activeAnnotationIdx,
    activeVertexIdx,
    annotations,
    selectedClassId,
    lockedBottleId,
    activeCameraId,
    hasUnsavedChanges
  ]);

  // ---------------------------------------------------------------------------
  // 5. Canvas Zoom & Pan Mouse Wheel / Drag
  // ---------------------------------------------------------------------------

  const handleWheel = (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.88;
    setZoom(prev => Math.max(0.8, Math.min(5.0, prev * zoomFactor)));
  };

  const handleMouseDownCanvas = (e) => {
    // Check if middle click or Pan tool or spacebar held
    if (e.button === 1 || activeTool === 'pan' || isSpacePressed) {
      e.preventDefault();
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
      return;
    }

    if (e.button !== 0) return; // Only left click for drawing

    const norm = getNormalizedCoords(e);
    if (!norm) return;

    if (activeTool === 'polygon') {
      // Check if clicking near first draft point to close loop
      if (draftPoints.length >= 3) {
        const first = draftPoints[0];
        const dist = Math.hypot(norm.x - first[0], norm.y - first[1]);
        if (dist < 0.04) {
          // Close Polygon Loop
          closeDraftPolygon();
          return;
        }
      }
      setDraftPoints(prev => [...prev, [norm.x, norm.y]]);
      setHasUnsavedChanges(true);
    } else if (activeTool === 'box') {
      setBoxStartPoint([norm.x, norm.y]);
      setBoxCurrentPoint([norm.x, norm.y]);
    }
  };

  const handleMouseMoveCanvas = (e) => {
    const norm = getNormalizedCoords(e);
    if (norm) {
      setCursorNormCoord(norm);
    }

    if (isPanning) {
      setPan({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y
      });
      return;
    }

    if (isDraggingVertex && activeAnnotationIdx !== null && activeVertexIdx !== null && norm) {
      // Dragging a vertex handle
      setAnnotations(prev => {
        const next = [...prev];
        const targetAnn = { ...next[activeAnnotationIdx] };
        const updatedPoints = [...targetAnn.points];
        updatedPoints[activeVertexIdx] = [norm.x, norm.y];
        targetAnn.points = updatedPoints;
        next[activeAnnotationIdx] = targetAnn;
        return next;
      });
      setHasUnsavedChanges(true);
      return;
    }

    if (activeTool === 'box' && boxStartPoint && norm) {
      setBoxCurrentPoint([norm.x, norm.y]);
    }
  };

  const handleMouseUpCanvas = () => {
    if (isPanning) {
      setIsPanning(false);
    }

    if (isDraggingVertex) {
      setIsDraggingVertex(false);
    }

    if (activeTool === 'box' && boxStartPoint && boxCurrentPoint) {
      const x1 = Math.min(boxStartPoint[0], boxCurrentPoint[0]);
      const x2 = Math.max(boxStartPoint[0], boxCurrentPoint[0]);
      const y1 = Math.min(boxStartPoint[1], boxCurrentPoint[1]);
      const y2 = Math.max(boxStartPoint[1], boxCurrentPoint[1]);

      // Only add if box has a minimum size
      if (Math.abs(x2 - x1) > 0.02 && Math.abs(y2 - y1) > 0.02) {
        const boxPolygon = [
          [x1, y1],
          [x2, y1],
          [x2, y2],
          [x1, y2]
        ];
        const currentClass = DEFECT_CLASSES[selectedClassId];
        const newAnn = {
          id: `${activeCameraId}_box_${Date.now()}`,
          class_id: currentClass.id,
          class_name: currentClass.name,
          points: boxPolygon
        };
        setAnnotations(prev => [...prev, newAnn]);
        setActiveAnnotationIdx(annotations.length);
        setHasUnsavedChanges(true);
        showNotification(`Created Bounding Box [${currentClass.name}]`, 'info');
      }

      setBoxStartPoint(null);
      setBoxCurrentPoint(null);
    }
  };

  const getNormalizedCoords = (e) => {
    if (!svgRef.current) return null;
    const rect = svgRef.current.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;

    const x = Math.max(0.0, Math.min(1.0, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0.0, Math.min(1.0, (e.clientY - rect.top) / rect.height));
    return { x: Number(x.toFixed(6)), y: Number(y.toFixed(6)) };
  };

  const closeDraftPolygon = () => {
    if (draftPoints.length < 3) return;
    const currentClass = DEFECT_CLASSES[selectedClassId];
    const newAnn = {
      id: `${activeCameraId}_poly_${Date.now()}`,
      class_id: currentClass.id,
      class_name: currentClass.name,
      points: draftPoints
    };
    setAnnotations(prev => [...prev, newAnn]);
    setActiveAnnotationIdx(annotations.length);
    setDraftPoints([]);
    setHasUnsavedChanges(true);
    showNotification(`Polygon loop closed & saved to draft [${currentClass.name}]`, 'info');
  };

  const handleDeleteVertex = (annIdx, vertIdx) => {
    setAnnotations(prev => {
      const next = [...prev];
      const target = { ...next[annIdx] };
      if (target.points.length <= 3) {
        // Less than 3 points would invalidate polygon; remove entire polygon
        return next.filter((_, i) => i !== annIdx);
      }
      target.points = target.points.filter((_, vi) => vi !== vertIdx);
      next[annIdx] = target;
      return next;
    });
    setActiveVertexIdx(null);
    setHasUnsavedChanges(true);
    showNotification('Vertex removed', 'info');
  };

  const handleDeleteAnnotation = (annIdx) => {
    setAnnotations(prev => prev.filter((_, i) => i !== annIdx));
    setActiveAnnotationIdx(null);
    setActiveVertexIdx(null);
    setHasUnsavedChanges(true);
    showNotification('Polygon contour deleted', 'info');
  };

  // ---------------------------------------------------------------------------
  // 6. Completeness Validation
  // ---------------------------------------------------------------------------

  const isLoopClosed = draftPoints.length === 0;
  const hasMinVertices = annotations.length > 0 && annotations.every(a => a.points && a.points.length >= 3);
  const hasValidClasses = annotations.length > 0 && annotations.every(a => a.class_id !== undefined && a.class_id !== null);
  const isBoundsValid = annotations.every(a => a.points.every(p => p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1));
  const isValidationPassing = annotations.length > 0 && hasMinVertices && hasValidClasses && isBoundsValid && isLoopClosed;

  // ---------------------------------------------------------------------------
  // 7. Commit & Save Ground Truth
  // ---------------------------------------------------------------------------

  const handleCommitAndSave = async () => {
    if (!lockedBottleId || !activeCameraId) return;

    if (!isValidationPassing) {
      showNotification('Cannot commit: Please complete all validation criteria first.', 'error');
      return;
    }

    setIsSaving(true);
    try {
      const activeClass = DEFECT_CLASSES[selectedClassId];
      const newVerdict = activeClass.id === 1 ? 'PASS' : 'REJECT';
      const newDefect = activeClass.id === 1 ? null : activeClass.name.toLowerCase().replace(' ', '_');

      const payload = {
        camera_id: activeCameraId,
        annotations: annotations,
        global_verdict: newVerdict,
        primary_defect: newDefect,
        verified: true
      };

      const res = await fetch(`/api/active-learning/bottle/${encodeURIComponent(lockedBottleId)}/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        setHasUnsavedChanges(false);
        showNotification(`Ground truth saved for Bottle #${lockedBottleId}! Database & YOLO labels updated.`, 'success');
        // Refresh candidate stats
        fetchCandidates(uncertainOnly, searchQuery);
      } else {
        const errData = await res.json().catch(() => ({}));
        showNotification(`Commit failed: ${errData.detail || 'Server error'}`, 'error');
      }
    } catch (err) {
      console.error('Error committing annotation:', err);
      showNotification('Network error while committing annotations.', 'error');
    } finally {
      setIsSaving(false);
    }
  };

  // ---------------------------------------------------------------------------
  // 8. Export YOLO Dataset Zip
  // ---------------------------------------------------------------------------

  const handleExportDataset = async () => {
    setIsExporting(true);
    try {
      const res = await fetch('/api/active-learning/export-dataset');
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `visionq_dataset_${Date.now()}.zip`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        showNotification('YOLOv11 dataset zip exported successfully!', 'success');
      } else {
        showNotification('Failed to generate dataset zip.', 'error');
      }
    } catch (err) {
      console.error('Error exporting dataset:', err);
      showNotification('Error exporting dataset.', 'error');
    } finally {
      setIsExporting(false);
    }
  };

  const showNotification = (text, type = 'info') => {
    setStatusMessage({ text, type });
    setTimeout(() => {
      setStatusMessage(prev => prev.text === text ? { text: '', type: 'info' } : prev);
    }, 4000);
  };

  const currentCameraView = bottleDetail?.cameras?.[activeCameraId];
  const currentBottleIdx = candidates.findIndex(c => c.bottle_id === lockedBottleId);

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] overflow-hidden bg-[#060e20] text-[#dae2fd] border border-[#1e2e48] rounded select-none">
      {/* =====================================================================
          TOP TELEMETRY NAVBAR & IMAGE SELECTOR
      ===================================================================== */}
      {/* =====================================================================
          TOP TELEMETRY NAVBAR & IMAGE SELECTOR
      ===================================================================== */}
      <header className="h-12 bg-[#0b1326] border-b border-[#1e2e48] px-3 flex items-center justify-between gap-2.5 shrink-0 z-30">
        {/* Left: Candidate Queue Selector */}
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <div className="flex items-center gap-1 shrink-0">
            <span className="font-['Outfit'] font-bold text-sm text-[#00f5a0] tracking-wide uppercase">VisionQ S3</span>
            <span className="hidden xl:inline-block font-mono text-[9px] px-1 py-0.5 bg-[#171f33] border border-[#1e2e48] text-[#849588]">STATION-ACTIVE</span>
          </div>

          <div className="h-4 w-px bg-[#1e2e48] shrink-0" />

          {/* Uncertain vs All Toggle */}
          <div className="flex bg-[#060e20] p-0.5 border border-[#1e2e48] rounded text-xs font-mono shrink-0">
            <button
              onClick={() => setUncertainOnly(true)}
              className={`px-1.5 xl:px-2 py-0.5 rounded transition-colors ${
                uncertainOnly
                  ? 'bg-[#171f33] text-[#00daf3] font-bold shadow-[0_0_8px_rgba(0,218,243,0.2)]'
                  : 'text-[#849588] hover:text-[#dae2fd]'
              }`}
              title="Filter by UNCERTAIN and borderline confidence (0.20 - 0.80)"
            >
              Uncertain ({uncertainCount})
            </button>
            <button
              onClick={() => setUncertainOnly(false)}
              className={`px-1.5 xl:px-2 py-0.5 rounded transition-colors ${
                !uncertainOnly
                  ? 'bg-[#171f33] text-[#00f5a0] font-bold shadow-[0_0_8px_rgba(0,245,160,0.2)]'
                  : 'text-[#849588] hover:text-[#dae2fd]'
              }`}
              title="Show all inspected bottles (re-evaluate any image)"
            >
              All ({totalCandidateCount})
            </button>
          </div>

          {/* Quick Search */}
          <div className="relative w-28 xl:w-40 shrink-0">
            <Search className="absolute left-2 top-2 text-[#849588]" size={12} />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search ID..."
              className="w-full bg-[#060e20] border border-[#1e2e48] pl-6 pr-2 py-1 rounded text-xs font-mono text-[#dae2fd] placeholder-[#849588] focus:outline-none focus:border-[#00f5a0]"
            />
          </div>

          {/* Candidate Dropdown Selector */}
          <div className="relative flex-1 min-w-[120px] max-w-[180px] xl:min-w-[200px] xl:max-w-sm">
            <select
              value={lockedBottleId || ''}
              onChange={(e) => handleRequestSwitchBottle(e.target.value)}
              className="w-full bg-[#060e20] border border-[#1e2e48] px-2 py-1 rounded text-xs font-mono text-[#00daf3] focus:outline-none focus:border-[#00f5a0] truncate cursor-pointer"
            >
              {candidates.map((cand) => (
                <option key={cand.bottle_id} value={cand.bottle_id}>
                  {cand.bottle_id} | {cand.global_verdict} {cand.primary_defect ? `(${cand.primary_defect})` : ''} - Conf:{' '}
                  {cand.primary_defect_confidence ? `${Math.round(cand.primary_defect_confidence * 100)}%` : 'N/A'}
                </option>
              ))}
            </select>
          </div>

          {/* Prev / Next Candidate Navigation */}
          <div className="flex items-center gap-0.5 shrink-0">
            <button
              disabled={currentBottleIdx <= 0}
              onClick={() => currentBottleIdx > 0 && handleRequestSwitchBottle(candidates[currentBottleIdx - 1].bottle_id)}
              className="p-1 rounded bg-[#060e20] border border-[#1e2e48] text-[#849588] hover:text-[#00f5a0] disabled:opacity-30 disabled:pointer-events-none"
              title="Previous Candidate"
            >
              <ChevronLeft size={13} />
            </button>
            <button
              disabled={currentBottleIdx >= candidates.length - 1}
              onClick={() => currentBottleIdx < candidates.length - 1 && handleRequestSwitchBottle(candidates[currentBottleIdx + 1].bottle_id)}
              className="p-1 rounded bg-[#060e20] border border-[#1e2e48] text-[#849588] hover:text-[#00f5a0] disabled:opacity-30 disabled:pointer-events-none"
              title="Next Candidate"
            >
              <ChevronRight size={13} />
            </button>
          </div>
        </div>

        {/* Center: Single-Image Lock Warning Banner */}
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded border text-[11px] font-mono font-bold tracking-wider shrink-0 transition-colors ${
          hasUnsavedChanges
            ? 'bg-[#ff3b30]/15 text-[#ff817b] border-[#ff3b30]/50 animate-pulse'
            : 'bg-[#171f33] text-[#00daf3] border-[#00daf3]/30'
        }`}>
          <Lock size={12} className={hasUnsavedChanges ? 'text-[#ff3b30]' : 'text-[#00daf3]'} />
          <span>
            {hasUnsavedChanges 
              ? (
                <>
                  <span className="xl:inline hidden">UNSAVED DRAFT: </span>
                  <span>#{lockedBottleId?.slice(-6) || '---'}</span>
                </>
              )
              : (
                <>
                  <span className="xl:inline hidden">LOCKED: </span>
                  <span>#{lockedBottleId?.slice(-6) || '---'}</span>
                </>
              )
            }
          </span>
        </div>

        {/* Right: Camera Switcher & Actions */}
        <div className="flex items-center gap-2 shrink-0">
          {/* Multi-Camera Angle Tabs */}
          {bottleDetail?.cameras && (
            <div className="flex bg-[#060e20] p-0.5 border border-[#1e2e48] rounded text-xs font-mono">
              {Object.keys(bottleDetail.cameras).map((camKey, idx) => (
                <button
                  key={camKey}
                  onClick={() => handleSwitchCamera(camKey)}
                  className={`px-2 py-0.5 rounded transition-colors flex items-center gap-1 ${
                    activeCameraId === camKey
                      ? 'bg-[#171f33] text-[#00f5a0] font-bold border border-[#00f5a0]/40'
                      : 'text-[#849588] hover:text-[#dae2fd]'
                  }`}
                >
                  <Camera size={11} />
                  <span>CAM-{idx + 1}: {camKey.replace('camera_', '').toUpperCase()}</span>
                </button>
              ))}
            </div>
          )}

          {/* Export YOLO Dataset Button */}
          <button
            onClick={handleExportDataset}
            disabled={isExporting}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#171f33] border border-[#00daf3]/40 text-[#00daf3] hover:bg-[#00daf3]/10 font-mono text-xs transition-colors shrink-0"
            title="Download full YOLOv11 dataset zip archive with train images, labels, and data.yaml"
          >
            <Download size={13} className={isExporting ? 'animate-bounce' : ''} />
            <span className="xl:inline hidden">Export YOLO Dataset</span>
            <span className="xl:hidden inline">Export [YOLO]</span>
          </button>

          {/* Help Button */}
          <button
            onClick={() => setShowHelpModal(true)}
            className="p-1 rounded bg-[#171f33] border border-[#1e2e48] text-[#849588] hover:text-[#00f5a0] transition-colors shrink-0"
            title="Shortcuts & Protocol Guide (?)"
          >
            <HelpCircle size={14} />
          </button>
        </div>
      </header>

      {/* =====================================================================
          MAIN WORKBENCH: CENTER CANVAS + RIGHT TOOL RAIL
      ===================================================================== */}
      <div className="flex-1 flex min-h-0 overflow-hidden relative">
        {/* CENTER: Inspection Canvas */}
        <main className="flex-1 flex flex-col bg-[#050811] relative overflow-hidden">
          {/* Top Canvas Viewport HUD */}
          <div className="h-8 bg-[#0b1326]/90 backdrop-blur border-b border-[#1e2e48] px-3 flex items-center justify-between text-xs font-mono z-20">
            {/* Zoom & Viewport Controls */}
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-[#849588] uppercase">ZOOM:</span>
              <button
                onClick={() => setZoom(prev => Math.max(0.8, prev - 0.25))}
                className="w-5 h-5 flex items-center justify-center bg-[#171f33] border border-[#1e2e48] text-[#849588] hover:text-[#00f5a0] rounded"
                title="Zoom Out (-)"
              >
                -
              </button>
              <div className="flex bg-[#060e20] border border-[#1e2e48] divide-x divide-[#1e2e48] rounded text-[10px]">
                <button
                  onClick={() => setZoom(1.0)}
                  className={`px-1.5 py-0.5 ${zoom === 1.0 ? 'bg-[#171f33] text-[#00f5a0] font-bold' : 'text-[#849588] hover:text-[#dae2fd]'}`}
                >
                  100%
                </button>
                <button
                  onClick={() => setZoom(2.0)}
                  className={`px-1.5 py-0.5 ${zoom === 2.0 ? 'bg-[#171f33] text-[#00f5a0] font-bold' : 'text-[#849588] hover:text-[#dae2fd]'}`}
                >
                  200%
                </button>
                <button
                  onClick={() => setZoom(3.0)}
                  className={`px-1.5 py-0.5 ${zoom === 3.0 ? 'bg-[#171f33] text-[#00f5a0] font-bold' : 'text-[#849588] hover:text-[#dae2fd]'}`}
                >
                  300%
                </button>
              </div>
              <button
                onClick={() => setZoom(prev => Math.min(5.0, prev + 0.25))}
                className="w-5 h-5 flex items-center justify-center bg-[#171f33] border border-[#1e2e48] text-[#849588] hover:text-[#00f5a0] rounded"
                title="Zoom In (+)"
              >
                +
              </button>
              <button
                onClick={() => { setZoom(1.0); setPan({ x: 0, y: 0 }); }}
                className="px-2 py-0.5 bg-[#171f33] border border-[#1e2e48] text-[10px] text-[#849588] hover:text-[#dae2fd] rounded flex items-center gap-1"
                title="Reset View to 1:1"
              >
                <RotateCcw size={10} /> 1:1 Reset
              </button>

              <div className="h-3 w-px bg-[#1e2e48] mx-1" />

              {/* Crosshair Toggle */}
              <button
                onClick={() => setShowCrosshairs(prev => !prev)}
                className={`px-2 py-0.5 border rounded text-[10px] flex items-center gap-1 transition-colors ${
                  showCrosshairs
                    ? 'bg-[#00daf3]/10 border-[#00daf3]/40 text-[#00daf3]'
                    : 'bg-[#171f33] border-[#1e2e48] text-[#849588]'
                }`}
                title="Toggle Reticle Crosshairs"
              >
                <Crosshair size={11} /> Reticle [{showCrosshairs ? 'ON' : 'OFF'}]
              </button>
            </div>

            {/* Specimen Telemetry Pill */}
            <div className="flex items-center gap-3 text-[10px] text-[#849588]">
              <span>BOTTLE: <strong className="text-[#dae2fd]">{lockedBottleId || 'NONE'}</strong></span>
              <span>|</span>
              <span>CAMERA: <strong className="text-[#00daf3]">{activeCameraId || 'N/A'}</strong></span>
              <span>|</span>
              <span>RES: <strong className="text-[#dae2fd]">{currentCameraView ? `${currentCameraView.width}x${currentCameraView.height}` : '---'}</strong></span>
              <span>|</span>
              <span>SCALE: <strong className="text-[#00f5a0]">{(zoom * 100).toFixed(0)}%</strong></span>
            </div>
          </div>

          {/* Interactive Canvas Viewport */}
          <div
            ref={canvasRef}
            onWheel={handleWheel}
            onMouseDown={handleMouseDownCanvas}
            onMouseMove={handleMouseMoveCanvas}
            onMouseUp={handleMouseUpCanvas}
            className={`flex-1 relative overflow-hidden flex items-center justify-center p-4 ${
              activeTool === 'pan' || isSpacePressed ? 'cursor-grab active:cursor-grabbing' : 'cursor-crosshair'
            }`}
            style={{
              backgroundImage: 'radial-gradient(circle, #1a2538 1px, transparent 1px)',
              backgroundSize: '24px 24px'
            }}
          >
            {/* Axis Reticle Crosshairs */}
            {showCrosshairs && (
              <>
                <div className="absolute inset-x-0 top-1/2 h-px border-t border-dashed border-[#1e2e48] pointer-events-none z-10" />
                <div className="absolute inset-y-0 left-1/2 w-px border-l border-dashed border-[#1e2e48] pointer-events-none z-10" />
              </>
            )}

            {/* Top-Left Coordinate HUD */}
            <div className="absolute top-3 left-3 bg-[#0b1326]/85 backdrop-blur border border-[#1e2e48] p-2 rounded text-[10px] font-mono pointer-events-none z-20 flex flex-col gap-0.5">
              <div className="text-[#849588] tracking-wider font-bold">NORMALIZED COORD HUD</div>
              <div className="text-[#dae2fd]">
                Cursor: <span className="text-[#00f5a0] font-bold">X: {cursorNormCoord.x.toFixed(4)}, Y: {cursorNormCoord.y.toFixed(4)}</span>
              </div>
              <div className="text-[#dae2fd]">
                Active Polygon: <span className="text-[#00daf3] font-bold">{annotations.length} items</span>
              </div>
            </div>

            {/* Interactive Image & SVG Specimen Container with Pan & Zoom */}
            <div
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: 'center center',
                transition: isPanning ? 'none' : 'transform 0.05s ease-out'
              }}
              className="relative max-w-full max-h-full flex items-center justify-center shadow-2xl border border-[#1e2e48]"
            >
              {/* Pristine Clean Image (Unannotated from Server) */}
              {currentCameraView?.image_url ? (
                <img
                  src={currentCameraView.image_url}
                  alt={`Bottle ${lockedBottleId} - ${activeCameraId}`}
                  className="max-h-[72vh] max-w-[70vw] object-contain block pointer-events-none select-none"
                  onDragStart={(e) => e.preventDefault()}
                />
              ) : (
                <div className="w-[500px] h-[500px] bg-[#0b1326] flex items-center justify-center font-mono text-xs text-[#849588]">
                  {isLoadingBottle ? 'Loading pristine bottle frame...' : 'No frame available'}
                </div>
              )}

              {/* Scalable SVG Polygon & Vertex Overlay */}
              <svg
                ref={svgRef}
                viewBox="0 0 1 1"
                preserveAspectRatio="none"
                className="absolute inset-0 w-full h-full z-10"
              >
                {/* 1. Existing Saved Polygons */}
                {annotations.map((ann, annIdx) => {
                  const classObj = DEFECT_CLASSES[ann.class_id] || DEFECT_CLASSES[0];
                  const pointsStr = ann.points.map(p => `${p[0]},${p[1]}`).join(' ');
                  const isSelected = activeAnnotationIdx === annIdx;

                  return (
                    <g key={ann.id || annIdx}>
                      {/* Polygon Surface */}
                      <polygon
                        points={pointsStr}
                        fill={classObj.color}
                        fillOpacity={isSelected ? 0.32 : 0.20}
                        stroke={classObj.color}
                        strokeWidth={0.003 / zoom}
                        strokeDasharray={isSelected ? `${0.006 / zoom},${0.003 / zoom}` : 'none'}
                        className="cursor-pointer transition-all hover:fill-opacity-40"
                        onClick={(e) => {
                          e.stopPropagation();
                          setActiveAnnotationIdx(annIdx);
                          setSelectedClassId(ann.class_id);
                        }}
                      />

                      {/* Vertex Nodes for Selected Polygon */}
                      {isSelected && ann.points.map((pt, vi) => (
                        <circle
                          key={vi}
                          cx={pt[0]}
                          cy={pt[1]}
                          r={(activeVertexIdx === vi ? 0.012 : 0.008) / zoom}
                          fill={activeVertexIdx === vi ? '#ffffff' : '#060e20'}
                          stroke={classObj.color}
                          strokeWidth={0.0025 / zoom}
                          className="cursor-grab active:cursor-grabbing hover:scale-125 transition-transform"
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            setActiveAnnotationIdx(annIdx);
                            setActiveVertexIdx(vi);
                            setIsDraggingVertex(true);
                          }}
                        />
                      ))}
                    </g>
                  );
                })}

                {/* 2. In-Progress Drafting Polygon */}
                {draftPoints.length > 0 && (
                  <g>
                    {/* Polyline / open polygon */}
                    <polyline
                      points={draftPoints.map(p => `${p[0]},${p[1]}`).join(' ')}
                      fill="none"
                      stroke={DEFECT_CLASSES[selectedClassId].color}
                      strokeWidth={0.0035 / zoom}
                      strokeDasharray={`${0.005 / zoom},${0.003 / zoom}`}
                    />

                    {/* Rubberband to cursor */}
                    <line
                      x1={draftPoints[draftPoints.length - 1][0]}
                      y1={draftPoints[draftPoints.length - 1][1]}
                      x2={cursorNormCoord.x}
                      y2={cursorNormCoord.y}
                      stroke={DEFECT_CLASSES[selectedClassId].color}
                      strokeWidth={0.002 / zoom}
                      strokeDasharray={`${0.003 / zoom},${0.003 / zoom}`}
                    />

                    {/* Draft Point Handles */}
                    {draftPoints.map((pt, dpi) => (
                      <circle
                        key={dpi}
                        cx={pt[0]}
                        cy={pt[1]}
                        r={(dpi === 0 ? 0.012 : 0.007) / zoom}
                        fill={dpi === 0 ? '#00f5a0' : '#060e20'}
                        stroke={DEFECT_CLASSES[selectedClassId].color}
                        strokeWidth={0.0025 / zoom}
                        className={dpi === 0 && draftPoints.length >= 3 ? 'cursor-pointer animate-ping' : ''}
                      />
                    ))}
                  </g>
                )}

                {/* 3. In-Progress Bounding Box Drafting */}
                {activeTool === 'box' && boxStartPoint && boxCurrentPoint && (
                  <rect
                    x={Math.min(boxStartPoint[0], boxCurrentPoint[0])}
                    y={Math.min(boxStartPoint[1], boxCurrentPoint[1])}
                    width={Math.abs(boxCurrentPoint[0] - boxStartPoint[0])}
                    height={Math.abs(boxCurrentPoint[1] - boxStartPoint[1])}
                    fill={DEFECT_CLASSES[selectedClassId].color}
                    fillOpacity={0.25}
                    stroke={DEFECT_CLASSES[selectedClassId].color}
                    strokeWidth={0.003 / zoom}
                    strokeDasharray={`${0.006 / zoom},${0.003 / zoom}`}
                  />
                )}
              </svg>
            </div>

            {/* Bottom Canvas Loop Status Indicator */}
            <div className="absolute bottom-3 left-3 bg-[#0b1326]/90 border border-[#1e2e48] px-2.5 py-1 rounded text-[10px] font-mono flex items-center gap-2 z-20">
              <span className={`w-2 h-2 rounded-full ${isLoopClosed ? 'bg-[#00f5a0]' : 'bg-[#f59e0b] animate-ping'}`} />
              <span>
                Loop Status:{' '}
                <strong className={isLoopClosed ? 'text-[#00f5a0]' : 'text-[#f59e0b]'}>
                  {isLoopClosed ? 'CLOSED (READY)' : 'DRAFTING IN PROGRESS...'}
                </strong>
              </span>
              {draftPoints.length > 0 && (
                <>
                  <span className="text-[#1e2e48]">|</span>
                  <span className="text-[#00daf3]">Points: {draftPoints.length}</span>
                  <span className="text-[#849588]">(Click start node to close)</span>
                </>
              )}
            </div>

            {/* In-Canvas Notification Pill */}
            {statusMessage.text && (
              <div className={`absolute top-4 right-4 px-3 py-1.5 rounded font-mono text-xs border shadow-lg z-30 transition-all ${
                statusMessage.type === 'success'
                  ? 'bg-[#00f5a0]/15 text-[#00f5a0] border-[#00f5a0]/50'
                  : statusMessage.type === 'error'
                  ? 'bg-[#ff3b30]/15 text-[#ff817b] border-[#ff3b30]/50'
                  : 'bg-[#171f33] text-[#dae2fd] border-[#1e2e48]'
              }`}>
                {statusMessage.text}
              </div>
            )}
          </div>
        </main>

        {/* RIGHT: Standalone Tool Rail, Defect Classes, Validation Gate & Commit Button */}
        <aside className="w-80 bg-[#0b1326] border-l border-[#1e2e48] flex flex-col justify-between overflow-y-auto shrink-0 z-20">
          <div className="p-3.5 flex flex-col gap-3.5">
            {/* SECTION 1: Tool Selection Grid */}
            <div>
              <div className="flex items-center justify-between pb-1 border-b border-[#1e2e48] mb-2">
                <span className="text-[10px] font-mono text-[#849588] uppercase tracking-wider font-bold">
                  ANNOTATION TOOLKIT
                </span>
                <span className="text-[9px] font-mono text-[#00daf3]">SHORTCUTS ACTIVE</span>
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                {/* Polygon Tool (P) */}
                <button
                  onClick={() => { setActiveTool('polygon'); setDraftPoints([]); }}
                  className={`flex items-center justify-between p-2 rounded border text-left transition-all ${
                    activeTool === 'polygon'
                      ? 'bg-[#171f33] border-[#00f5a0] text-[#00f5a0] shadow-[0_0_8px_rgba(0,245,160,0.2)] font-semibold'
                      : 'bg-[#060e20] border-[#1e2e48] text-[#849588] hover:text-[#dae2fd]'
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <Layers size={14} />
                    <span className="text-xs font-mono">Polygon</span>
                  </div>
                  <span className="text-[9px] font-mono bg-[#171f33] px-1 py-0.2 rounded border border-[#1e2e48]">P</span>
                </button>

                {/* Bounding Box Tool (B) */}
                <button
                  onClick={() => { setActiveTool('box'); setDraftPoints([]); }}
                  className={`flex items-center justify-between p-2 rounded border text-left transition-all ${
                    activeTool === 'box'
                      ? 'bg-[#171f33] border-[#00f5a0] text-[#00f5a0] shadow-[0_0_8px_rgba(0,245,160,0.2)] font-semibold'
                      : 'bg-[#060e20] border-[#1e2e48] text-[#849588] hover:text-[#dae2fd]'
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <Crop size={14} />
                    <span className="text-xs font-mono">Box</span>
                  </div>
                  <span className="text-[9px] font-mono bg-[#171f33] px-1 py-0.2 rounded border border-[#1e2e48]">B</span>
                </button>

                {/* Select / Move Node (V) */}
                <button
                  onClick={() => { setActiveTool('move'); setDraftPoints([]); }}
                  className={`flex items-center justify-between p-2 rounded border text-left transition-all ${
                    activeTool === 'move'
                      ? 'bg-[#171f33] border-[#00f5a0] text-[#00f5a0] shadow-[0_0_8px_rgba(0,245,160,0.2)] font-semibold'
                      : 'bg-[#060e20] border-[#1e2e48] text-[#849588] hover:text-[#dae2fd]'
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <Move size={14} />
                    <span className="text-xs font-mono">Move Node</span>
                  </div>
                  <span className="text-[9px] font-mono bg-[#171f33] px-1 py-0.2 rounded border border-[#1e2e48]">V</span>
                </button>

                {/* Pan Hand (H) */}
                <button
                  onClick={() => { setActiveTool('pan'); setDraftPoints([]); }}
                  className={`flex items-center justify-between p-2 rounded border text-left transition-all ${
                    activeTool === 'pan'
                      ? 'bg-[#171f33] border-[#00f5a0] text-[#00f5a0] shadow-[0_0_8px_rgba(0,245,160,0.2)] font-semibold'
                      : 'bg-[#060e20] border-[#1e2e48] text-[#849588] hover:text-[#dae2fd]'
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <Hand size={14} />
                    <span className="text-xs font-mono">Pan View</span>
                  </div>
                  <span className="text-[9px] font-mono bg-[#171f33] px-1 py-0.2 rounded border border-[#1e2e48]">H</span>
                </button>

                {/* Delete Vertex / Contour (Del) */}
                <button
                  onClick={() => {
                    if (activeAnnotationIdx !== null) {
                      if (activeVertexIdx !== null) {
                        handleDeleteVertex(activeAnnotationIdx, activeVertexIdx);
                      } else {
                        handleDeleteAnnotation(activeAnnotationIdx);
                      }
                    }
                  }}
                  disabled={activeAnnotationIdx === null}
                  className="flex items-center justify-between p-2 rounded border border-[#1e2e48] bg-[#060e20] text-[#849588] hover:text-[#ff3b30] hover:border-[#ff3b30]/40 text-left transition-all disabled:opacity-30 disabled:pointer-events-none"
                  title="Delete Selected Node or Contour (Del)"
                >
                  <div className="flex items-center gap-1.5">
                    <Trash2 size={14} />
                    <span className="text-xs font-mono">Delete</span>
                  </div>
                  <span className="text-[9px] font-mono bg-[#171f33] px-1 py-0.2 rounded border border-[#1e2e48]">DEL</span>
                </button>

                {/* Close Loop Action */}
                <button
                  onClick={closeDraftPolygon}
                  disabled={draftPoints.length < 3}
                  className="flex items-center justify-between p-2 rounded border border-[#00f5a0]/40 bg-[#00f5a0]/10 text-[#00f5a0] hover:bg-[#00f5a0]/20 text-left transition-all disabled:opacity-30 disabled:pointer-events-none font-semibold"
                  title="Close current open polygon loop"
                >
                  <div className="flex items-center gap-1.5">
                    <CheckCircle2 size={14} />
                    <span className="text-xs font-mono">Close Loop</span>
                  </div>
                  <span className="text-[9px] font-mono bg-[#00f5a0]/20 px-1 py-0.2 rounded border border-[#00f5a0]/40">OK</span>
                </button>
              </div>
            </div>

            {/* SECTION 2: Defect Ground Truth Class Palette (1-6) */}
            <div>
              <div className="flex items-center justify-between pb-1 border-b border-[#1e2e48] mb-2">
                <span className="text-[10px] font-mono text-[#849588] uppercase tracking-wider font-bold">
                  GROUND TRUTH DEFECT CLASS
                </span>
                <span className="text-[9px] font-mono text-[#849588]">KEYS 1-6</span>
              </div>
              <div className="flex flex-col gap-1">
                {DEFECT_CLASSES.map((cls) => {
                  const isSelected = selectedClassId === cls.id;
                  return (
                    <button
                      key={cls.id}
                      onClick={() => {
                        setSelectedClassId(cls.id);
                        if (activeAnnotationIdx !== null && annotations[activeAnnotationIdx]) {
                          setAnnotations(prev => {
                            const next = [...prev];
                            next[activeAnnotationIdx] = {
                              ...next[activeAnnotationIdx],
                              class_id: cls.id,
                              class_name: cls.name
                            };
                            return next;
                          });
                          setHasUnsavedChanges(true);
                        }
                      }}
                      className={`flex items-center justify-between p-2 rounded border text-left transition-all ${
                        isSelected
                          ? `bg-[#171f33] ${cls.border} text-[#dae2fd] shadow-[0_0_8px_rgba(0,0,0,0.4)]`
                          : 'bg-[#060e20] border-[#1e2e48] text-[#849588] hover:border-[#849588]'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className="w-3 h-3 rounded-none shrink-0"
                          style={{ backgroundColor: cls.color }}
                        />
                        <span className={`text-xs font-mono ${isSelected ? 'font-bold text-[#dae2fd]' : 'text-[#849588]'}`}>
                          {cls.key}. {cls.name}
                        </span>
                      </div>
                      <span
                        className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded"
                        style={{
                          color: cls.color,
                          backgroundColor: `${cls.color}15`,
                          border: `1px solid ${cls.color}40`
                        }}
                      >
                        [{cls.key}]
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* SECTION 3: Geometry & Contour Metrics */}
            <div className="bg-[#060e20] border border-[#1e2e48] p-2.5 rounded">
              <div className="flex items-center justify-between pb-1 mb-2 border-b border-[#1e2e48]">
                <span className="text-[10px] font-mono text-[#849588] uppercase tracking-wider font-bold">
                  GEOMETRY METRICS
                </span>
                <Sliders size={12} className="text-[#849588]" />
              </div>
              <div className="grid grid-cols-2 gap-1.5 text-xs font-mono">
                <div className="bg-[#0b1326] p-1.5 rounded border border-[#1e2e48]">
                  <div className="text-[9px] text-[#849588]">CONTOUR COUNT</div>
                  <div className="text-sm font-bold text-[#00daf3] mt-0.5">{annotations.length} items</div>
                </div>
                <div className="bg-[#0b1326] p-1.5 rounded border border-[#1e2e48]">
                  <div className="text-[9px] text-[#849588]">ACTIVE VERTICES</div>
                  <div className="text-sm font-bold text-[#00f5a0] mt-0.5">
                    {activeAnnotationIdx !== null && annotations[activeAnnotationIdx]
                      ? `${annotations[activeAnnotationIdx].points.length} nodes`
                      : '---'}
                  </div>
                </div>
              </div>
            </div>

            {/* SECTION 4: Validation Checklist */}
            <div className="bg-[#060e20] border border-[#1e2e48] p-2.5 rounded">
              <div className="flex items-center justify-between pb-1 mb-2 border-b border-[#1e2e48]">
                <span className="text-[10px] font-mono text-[#849588] uppercase tracking-wider font-bold">
                  VALIDATION CHECKLIST
                </span>
                <span className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded border ${
                  isValidationPassing
                    ? 'bg-[#00f5a0]/15 text-[#00f5a0] border-[#00f5a0]/40'
                    : 'bg-[#f59e0b]/15 text-[#f59e0b] border-[#f59e0b]/40'
                }`}>
                  {isValidationPassing ? 'READY TO COMMIT' : 'INCOMPLETE'}
                </span>
              </div>
              <div className="flex flex-col gap-1.5 text-xs font-mono">
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-1.5 text-[#dae2fd]">
                    <CheckCircle2 size={12} className={hasMinVertices ? 'text-[#00f5a0]' : 'text-[#849588]'} />
                    Polygon Loop Integrity
                  </span>
                  <span className={hasMinVertices ? 'text-[#00f5a0] font-bold' : 'text-[#849588]'}>
                    {hasMinVertices ? 'VALID (OK)' : 'NO CONTOUR'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-1.5 text-[#dae2fd]">
                    <CheckCircle2 size={12} className={hasValidClasses ? 'text-[#00f5a0]' : 'text-[#849588]'} />
                    Ground Truth Assigned
                  </span>
                  <span className="text-[#dae2fd] font-bold">
                    {DEFECT_CLASSES[selectedClassId]?.name || 'NONE'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-1.5 text-[#dae2fd]">
                    <CheckCircle2 size={12} className={isBoundsValid ? 'text-[#00f5a0]' : 'text-[#849588]'} />
                    Bounds Within Optical FOV
                  </span>
                  <span className={isBoundsValid ? 'text-[#00f5a0] font-bold' : 'text-[#ff3b30]'}>
                    {isBoundsValid ? '100% IN BOUNDS' : 'OUT OF BOUNDS'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* SECTION 5: Commit Actions Footer */}
          <div className="p-3 border-t border-[#1e2e48] bg-[#060e20] flex flex-col gap-2">
            <button
              onClick={handleCommitAndSave}
              disabled={isSaving || !isValidationPassing}
              className={`w-full py-2.5 px-3 rounded font-mono text-xs font-bold flex items-center justify-center gap-2 border transition-all active:scale-[0.98] ${
                isValidationPassing
                  ? 'bg-[#00f5a0] hover:bg-[#00daf3] text-[#050811] border-[#00f5a0] shadow-[0_0_12px_rgba(0,245,160,0.3)] cursor-pointer'
                  : 'bg-[#171f33] text-[#849588] border-[#1e2e48] opacity-60 cursor-not-allowed'
              }`}
            >
              <Save size={14} className={isSaving ? 'animate-spin' : ''} />
              <span>{isSaving ? 'Committing Truth...' : 'Commit & Save Truth [Ctrl+S]'}</span>
            </button>

            <button
              onClick={handleDiscardCurrentLock}
              className="w-full py-1.5 px-3 rounded bg-[#171f33] hover:bg-[#1e2e48] text-[#849588] hover:text-[#ff3b30] border border-[#1e2e48] font-mono text-xs flex items-center justify-center gap-1.5 transition-colors"
            >
              <Unlock size={12} />
              <span>Discard & Release Lock</span>
            </button>
          </div>
        </aside>
      </div>

      {/* =====================================================================
          BOTTOM STATUS & HOTKEY LEGEND DOCK
      ===================================================================== */}
      <footer className="h-7 bg-[#0b1326] border-t border-[#1e2e48] px-3 flex items-center justify-between text-[11px] font-mono shrink-0 z-30">
        {/* Hotkey Cheatsheet Chips */}
        <div className="flex items-center gap-1.5 overflow-x-auto text-[#849588] whitespace-nowrap">
          <span className="text-[#dae2fd] font-bold mr-1">HOTKEYS:</span>
          <span className="inline-flex items-center gap-1">
            <strong className="bg-[#171f33] border border-[#1e2e48] px-1 py-0.2 rounded text-[#dae2fd]">P</strong> Poly
          </span>
          <span>·</span>
          <span className="inline-flex items-center gap-1">
            <strong className="bg-[#171f33] border border-[#1e2e48] px-1 py-0.2 rounded text-[#dae2fd]">B</strong> Box
          </span>
          <span>·</span>
          <span className="inline-flex items-center gap-1">
            <strong className="bg-[#171f33] border border-[#1e2e48] px-1 py-0.2 rounded text-[#dae2fd]">V</strong> Move
          </span>
          <span>·</span>
          <span className="inline-flex items-center gap-1">
            <strong className="bg-[#171f33] border border-[#1e2e48] px-1 py-0.2 rounded text-[#dae2fd]">1-6</strong> Class
          </span>
          <span>·</span>
          <span className="inline-flex items-center gap-1">
            <strong className="bg-[#171f33] border border-[#00f5a0]/40 px-1 py-0.2 rounded text-[#00f5a0]">Ctrl+S</strong> Save
          </span>
          <span>·</span>
          <span className="inline-flex items-center gap-1">
            <strong className="bg-[#171f33] border border-[#1e2e48] px-1 py-0.2 rounded text-[#dae2fd]">Del</strong> Erase
          </span>
          <span className="hidden xl:inline">·</span>
          <span className="hidden xl:inline-flex items-center gap-1">
            <strong className="bg-[#171f33] border border-[#1e2e48] px-1 py-0.2 rounded text-[#dae2fd]">H</strong> Pan
          </span>
          <span className="hidden 2xl:inline">·</span>
          <span
            onClick={() => setShowHelpModal(true)}
            className="hidden 2xl:inline-flex items-center gap-1 text-[#00daf3] hover:underline cursor-pointer"
          >
            <strong className="bg-[#171f33] border border-[#00daf3]/40 px-1 py-0.2 rounded">?</strong> Guide
          </span>
        </div>

        {/* Retraining Buffer Progress */}
        <div className="flex items-center gap-2.5 shrink-0 whitespace-nowrap">
          <div className="flex items-center gap-1">
            <span className="text-[#849588] hidden md:inline">Verified:</span>
            <span className="text-[#00f5a0] font-bold">
              {totalCandidateCount - uncertainCount}/{totalCandidateCount}
            </span>
          </div>
          <div className="hidden lg:block h-3 w-px bg-[#1e2e48]" />
          <div className="hidden lg:flex items-center gap-1 text-[#849588]">
            <span className="w-1.5 h-1.5 rounded-full bg-[#00f5a0] animate-pulse" />
            <span className="hidden xl:inline">ACTIVE RETRAINING READY</span>
            <span className="xl:hidden inline">ACTIVE</span>
          </div>
        </div>
      </footer>

      {/* =====================================================================
          SINGLE-IMAGE LOCK GUARD CONFIRMATION MODAL
      ===================================================================== */}
      {showLockGuardModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-[#0b1326] border border-[#ff3b30]/50 rounded-lg p-5 max-w-md w-full shadow-2xl flex flex-col gap-3 font-mono">
            <div className="flex items-center gap-2 text-[#ff817b]">
              <ShieldAlert size={22} className="text-[#ff3b30]" />
              <h3 className="font-['Outfit'] font-bold text-base text-[#dae2fd]">
                Active Image Locked for Editing
              </h3>
            </div>
            <p className="text-xs text-[#849588] leading-relaxed">
              Bottle <strong className="text-[#00daf3]">#{lockedBottleId}</strong> is currently locked with uncommitted annotation changes. To ensure dataset authenticity and avoid accidental overwrites, only one inspection can be edited at a time.
            </p>
            <p className="text-xs text-[#ff817b] bg-[#ff3b30]/10 p-2 rounded border border-[#ff3b30]/30">
              Please commit your verified annotations or discard the draft before switching to another inspection.
            </p>
            <div className="flex items-center justify-end gap-2 mt-2">
              <button
                onClick={() => setShowLockGuardModal(false)}
                className="px-3 py-1.5 rounded bg-[#171f33] text-[#dae2fd] text-xs hover:bg-[#1e2e48]"
              >
                Stay on Current Bottle
              </button>
              <button
                onClick={handleConfirmDiscardAndSwitch}
                className="px-3 py-1.5 rounded bg-[#ff3b30]/20 text-[#ff817b] border border-[#ff3b30]/50 text-xs hover:bg-[#ff3b30]/30 font-bold"
              >
                Discard & Switch
              </button>
            </div>
          </div>
        </div>
      )}

      {/* =====================================================================
          HELP & PROTOCOL GUIDE MODAL
      ===================================================================== */}
      {showHelpModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-[#0b1326] border border-[#1e2e48] rounded-lg p-5 max-w-xl w-full shadow-2xl flex flex-col gap-4 font-mono max-h-[85vh] overflow-y-auto">
            <div className="flex items-center justify-between pb-2 border-b border-[#1e2e48]">
              <div className="flex items-center gap-2">
                <HelpCircle size={18} className="text-[#00daf3]" />
                <h3 className="font-['Outfit'] font-bold text-base text-[#00daf3]">
                  Active Learning & Annotation Station Protocol
                </h3>
              </div>
              <button
                onClick={() => setShowHelpModal(false)}
                className="text-[#849588] hover:text-[#dae2fd] p-1 rounded"
              >
                <X size={16} />
              </button>
            </div>

            <div className="flex flex-col gap-3 text-xs text-[#849588] leading-relaxed">
              <div>
                <h4 className="font-bold text-[#dae2fd] mb-1">1. Active Learning Candidate Queue</h4>
                <p>
                  By default, the station displays ambiguous and edge-case specimens (`UNCERTAIN` or confidence between 20% and 80%). You can toggle to "All Images" or use the search bar to re-evaluate and re-annotate any bottle in the historical inspection archive.
                </p>
              </div>

              <div>
                <h4 className="font-bold text-[#dae2fd] mb-1">2. Single-Image Lock Integrity</h4>
                <p>
                  To maintain data durability and single-editor audit integrity, only one bottle inspection can be opened in the editor at a time. The system will guard against accidental switches until the current annotation is committed or discarded.
                </p>
              </div>

              <div>
                <h4 className="font-bold text-[#dae2fd] mb-1">3. Annotation Tools & Keyboard Shortcuts</h4>
                <ul className="list-disc pl-5 flex flex-col gap-1 text-[#dae2fd]">
                  <li><strong className="text-[#00f5a0]">P</strong>: Polygon tool — click around the defect perimeter; click first node to close loop.</li>
                  <li><strong className="text-[#00f5a0]">B</strong>: Bounding Box tool — click and drag rectangle.</li>
                  <li><strong className="text-[#00f5a0]">V</strong>: Move Node tool — click and drag vertex handles to refine contour edge.</li>
                  <li><strong className="text-[#00f5a0]">H</strong> or <strong className="text-[#00f5a0]">Hold Space</strong>: Pan view across the high-resolution frame.</li>
                  <li><strong className="text-[#00f5a0]">Mouse Wheel</strong>: Continuous zoom from 80% to 500%.</li>
                  <li><strong className="text-[#00f5a0]">1 to 6</strong>: Switch ground truth defect class (Damaged, Good, Misplaced, No Cap, Open, Wet).</li>
                  <li><strong className="text-[#00f5a0]">Del / Backspace</strong>: Remove selected vertex node or entire contour.</li>
                  <li><strong className="text-[#00f5a0]">Ctrl + S</strong>: Validate and commit ground truth to database and YOLO label format.</li>
                </ul>
              </div>

              <div>
                <h4 className="font-bold text-[#dae2fd] mb-1">4. Validation Gate & Continuous Retraining</h4>
                <p>
                  The commit button verifies that all polygons have closed loops, valid classes, and are within bounds. When committed, labels are saved in normalized YOLO segmentation format, SQLite database is updated, and the dataset can be exported anytime via "Export YOLO Dataset".
                </p>
              </div>
            </div>

            <div className="flex justify-end pt-2 border-t border-[#1e2e48]">
              <button
                onClick={() => setShowHelpModal(false)}
                className="px-4 py-1.5 rounded bg-[#00f5a0] text-[#050811] font-bold text-xs"
              >
                Close & Return to Editor
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
