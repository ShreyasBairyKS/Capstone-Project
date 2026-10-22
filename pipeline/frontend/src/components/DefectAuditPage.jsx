import React, { useState, useEffect, useCallback } from 'react';
import { 
  Layers, 
  Search, 
  Filter, 
  Download, 
  RefreshCw, 
  Eye, 
  Copy, 
  Check, 
  ChevronLeft, 
  ChevronRight,
  ShieldAlert, 
  ShieldCheck, 
  AlertTriangle, 
  Clock,
  CheckCircle2,
  AlertOctagon
} from 'lucide-react';

export default function DefectAuditPage({ onSelectBottle }) {
  const [records, setRecords] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [copiedId, setCopiedId] = useState(null);

  // Filter States
  const [verdict, setVerdict] = useState('REJECT');
  const [defectType, setDefectType] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [timePreset, setTimePreset] = useState('all'); // '15m', '1h', '8h', '24h', 'all'

  // Pagination States
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);

  const defectClasses = [
    { id: '', label: 'ALL DEFECT CLASSES' },
    { id: 'damaged_cap', label: 'Damaged Cap' },
    { id: 'misplaced_cap', label: 'Misplaced / Skewed Cap' },
    { id: 'no_cap', label: 'Missing Cap' },
    { id: 'open_cap', label: 'Open / Unsealed Cap' },
    { id: 'wet_cap', label: 'Wet Cap / Condensation' },
  ];

  // Calculate epoch range based on preset
  const getEpochRange = useCallback(() => {
    const now = Date.now() / 1000;
    switch (timePreset) {
      case '15m': return { start: now - 15 * 60, end: now };
      case '1h':  return { start: now - 60 * 60, end: now };
      case '8h':  return { start: now - 8 * 60 * 60, end: now };
      case '24h': return { start: now - 24 * 60 * 60, end: now };
      default:    return { start: null, end: null };
    }
  }, [timePreset]);

  // Fetch paginated records from backend
  const fetchRecords = useCallback(async () => {
    setLoading(true);
    try {
      const { start, end } = getEpochRange();
      const params = new URLSearchParams({
        limit: pageSize.toString(),
        offset: (page * pageSize).toString(),
        paginate: 'true'
      });

      if (verdict && verdict !== 'ALL') params.append('verdict', verdict);
      if (defectType) params.append('defect_type', defectType);
      if (searchTerm.trim()) params.append('search', searchTerm.trim());
      if (start) params.append('start_epoch', start.toString());
      if (end) params.append('end_epoch', end.toString());

      const res = await fetch(`/api/defects?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        if (data.items) {
          setRecords(data.items);
          setTotalCount(data.total_count);
        } else if (Array.isArray(data)) {
          setRecords(data);
          setTotalCount(data.length);
        }
      }
    } catch (err) {
      console.error('Failed to load audit records:', err);
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, verdict, defectType, searchTerm, getEpochRange]);

  useEffect(() => {
    fetchRecords();
  }, [fetchRecords]);

  // Reset to page 0 whenever filter conditions change
  const handleFilterChange = (setter, val) => {
    setter(val);
    setPage(0);
  };

  // CSV Export Handler
  const handleExportCSV = async () => {
    setExporting(true);
    try {
      const { start, end } = getEpochRange();
      const params = new URLSearchParams({ limit: '10000' });
      if (verdict && verdict !== 'ALL') params.append('verdict', verdict);
      if (defectType) params.append('defect_type', defectType);
      if (searchTerm.trim()) params.append('search', searchTerm.trim());
      if (start) params.append('start_epoch', start.toString());
      if (end) params.append('end_epoch', end.toString());

      const res = await fetch(`/api/audit/export?${params.toString()}`);
      if (!res.ok) throw new Error('Export failed');

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const ts = new Date().toISOString().replace(/[:.]/g, '-');
      a.download = `visionq_defect_audit_${ts}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('CSV Export error:', err);
    } finally {
      setExporting(false);
    }
  };

  const copyToClipboard = (id) => {
    navigator.clipboard.writeText(id);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const totalPages = Math.ceil(totalCount / pageSize) || 1;

  return (
    <div className="flex flex-col gap-3.5 w-full">
      {/* Top Banner / Summary Strip */}
      <div className="bg-[#0b1326] dark:bg-[#060e20] light:bg-[#ffffff] border border-[#1e2e48] light:border-[#e2e8f0] p-4 rounded shadow-sm flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <Layers className="text-[#00daf3]" size={22} />
            <h2 className="font-['Outfit'] font-bold text-xl text-[#dae2fd] light:text-[#0f172a] tracking-tight">
              Defect Audit & Historical Traceability
            </h2>
          </div>
          <p className="text-xs font-mono text-[#849588] mt-1">
            ISO 9001 Compliance Log • Continuous SQLite WAL Serialization & Actuator Tracking
          </p>
        </div>

        {/* Micro-Telemetry Cards */}
        <div className="flex items-center gap-3">
          <div className="px-3 py-1.5 rounded bg-[#171f33] border border-[#1e2e48] text-right font-mono">
            <div className="text-[10px] text-[#849588] uppercase">FILTERED EVENTS</div>
            <div className="text-base font-bold text-[#00daf3]">{totalCount.toLocaleString()}</div>
          </div>
          <div className="px-3 py-1.5 rounded bg-[#171f33] border border-[#1e2e48] text-right font-mono">
            <div className="text-[10px] text-[#849588] uppercase">DATABASE MODE</div>
            <div className="text-xs font-bold text-[#00f5a0]">SQLITE WAL (NORMAL)</div>
          </div>
        </div>
      </div>

      {/* Filter & Controls Toolbar */}
      <div className="bg-[#0b1326] dark:bg-[#060e20] light:bg-[#ffffff] border border-[#1e2e48] light:border-[#e2e8f0] p-3 rounded shadow-sm flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          {/* Verdict Filter Tabs */}
          <div className="flex items-center gap-1 bg-[#171f33] p-1 rounded border border-[#1e2e48] text-xs font-mono">
            {['ALL', 'REJECT', 'UNCERTAIN', 'PASS'].map((v) => (
              <button
                key={v}
                onClick={() => handleFilterChange(setVerdict, v)}
                className={`px-3 py-1 rounded transition-colors font-semibold
                  ${verdict === v 
                    ? 'bg-[#00daf3] text-[#050811] shadow' 
                    : 'text-[#94a3b8] hover:text-[#dae2fd] hover:bg-[#1f2c47]'}`
                }
              >
                {v === 'REJECT' && <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#ff3b30] mr-1.5" />}
                {v === 'UNCERTAIN' && <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#f59e0b] mr-1.5" />}
                {v === 'PASS' && <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#00f5a0] mr-1.5" />}
                {v}
              </button>
            ))}
          </div>

          {/* Time Window Presets */}
          <div className="flex items-center gap-1 text-xs font-mono">
            <span className="text-[#849588] text-[11px] mr-1">WINDOW:</span>
            {[
              { id: '15m', label: '15m' },
              { id: '1h', label: '1h' },
              { id: '8h', label: 'Shift (8h)' },
              { id: '24h', label: '24h' },
              { id: 'all', label: 'All Time' },
            ].map((p) => (
              <button
                key={p.id}
                onClick={() => handleFilterChange(setTimePreset, p.id)}
                className={`px-2 py-1 rounded border text-[11px] font-mono transition-colors
                  ${timePreset === p.id 
                    ? 'bg-[#171f33] text-[#00f5a0] border-[#00f5a0]/40 font-bold' 
                    : 'bg-transparent text-[#849588] border-[#1e2e48] hover:border-[#849588]'}`
                }
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Export CSV & Refresh Actions */}
          <div className="flex items-center gap-2">
            <button
              onClick={handleExportCSV}
              disabled={exporting || totalCount === 0}
              className="h-8 px-3 rounded bg-[#00f5a0] hover:bg-[#00e293] text-[#050811] text-xs font-mono font-bold flex items-center gap-1.5 transition-all shadow-sm disabled:opacity-50"
              title="Export all matching records to CSV"
            >
              <Download size={14} className={exporting ? 'animate-bounce' : ''} />
              <span>{exporting ? 'EXPORTING CSV...' : 'EXPORT CSV'}</span>
            </button>
            <button
              onClick={fetchRecords}
              disabled={loading}
              className="w-8 h-8 rounded bg-[#171f33] hover:bg-[#1f2c47] border border-[#1e2e48] flex items-center justify-center text-[#dae2fd] transition-colors"
              title="Refresh records"
            >
              <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            </button>
          </div>
        </div>

        {/* Second Row: Search and Defect Dropdown */}
        <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-[#1e2e48]/60">
          {/* Substring Search */}
          <div className="relative flex-1 min-w-[240px]">
            <Search size={14} className="absolute left-3 top-2.5 text-[#849588]" />
            <input
              type="text"
              placeholder="Filter by Bottle ID (e.g. BOTTLE_20261017)..."
              value={searchTerm}
              onChange={(e) => handleFilterChange(setSearchTerm, e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 bg-[#060e20] border border-[#1e2e48] rounded text-xs font-mono text-[#dae2fd] placeholder-[#64748b] focus:outline-none focus:border-[#00daf3]"
            />
          </div>

          {/* Defect Class Dropdown */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-[#849588]">DEFECT:</span>
            <select
              value={defectType}
              onChange={(e) => handleFilterChange(setDefectType, e.target.value)}
              className="bg-[#060e20] border border-[#1e2e48] text-xs font-mono text-[#dae2fd] rounded px-3 py-1.5 focus:outline-none focus:border-[#00daf3]"
            >
              {defectClasses.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* High-Density Data Grid */}
      <div className="bg-[#0b1326] dark:bg-[#060e20] light:bg-[#ffffff] border border-[#1e2e48] light:border-[#e2e8f0] rounded overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono border-collapse min-w-[900px]">
            <thead>
              <tr className="bg-[#060e20] border-b border-[#1e2e48] text-[#849588] text-[10px] tracking-wider uppercase">
                <th className="py-2.5 px-3">BOTTLE ID</th>
                <th className="py-2.5 px-3">TIMESTAMP</th>
                <th className="py-2.5 px-3">VERDICT</th>
                <th className="py-2.5 px-3">PRIMARY DEFECT</th>
                <th className="py-2.5 px-3">CONFIDENCE</th>
                <th className="py-2.5 px-3">FLAGGED VIEWS</th>
                <th className="py-2.5 px-3">ACTUATOR</th>
                <th className="py-2.5 px-3">LATENCY</th>
                <th className="py-2.5 px-3 text-right">ACTIONS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1e2e48]/60">
              {loading && records.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-12 text-center text-[#849588]">
                    <div className="flex flex-col items-center gap-2">
                      <RefreshCw size={24} className="animate-spin text-[#00daf3]" />
                      <span>Loading historical audit records...</span>
                    </div>
                  </td>
                </tr>
              ) : records.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-12 text-center text-[#849588]">
                    <div className="flex flex-col items-center gap-1">
                      <Filter size={24} className="text-[#64748b]" />
                      <span className="font-bold text-[#dae2fd]">No audit records found matching active filter.</span>
                      <span className="text-[11px]">Try expanding the time window or clearing filters.</span>
                    </div>
                  </td>
                </tr>
              ) : (
                records.map((r) => {
                  const isPass = r.global_verdict === 'PASS';
                  const isReject = r.global_verdict === 'REJECT';
                  const isUncertain = r.global_verdict === 'UNCERTAIN';
                  const conf = r.primary_defect_confidence;

                  return (
                    <tr 
                      key={r.bottle_id} 
                      className="hover:bg-[#171f33]/60 transition-colors cursor-pointer"
                      onClick={() => onSelectBottle && onSelectBottle(r.bottle_id)}
                    >
                      {/* Bottle ID with Copy */}
                      <td className="py-2.5 px-3 font-bold text-[#00daf3]">
                        <div className="flex items-center gap-1.5">
                          <span>{r.bottle_id}</span>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              copyToClipboard(r.bottle_id);
                            }}
                            className="text-[#64748b] hover:text-[#00daf3] transition-colors p-0.5"
                            title="Copy Bottle ID"
                          >
                            {copiedId === r.bottle_id ? <Check size={12} className="text-[#00f5a0]" /> : <Copy size={12} />}
                          </button>
                        </div>
                      </td>

                      {/* Timestamp */}
                      <td className="py-2.5 px-3 text-[#94a3b8] whitespace-nowrap">
                        {r.timestamp ? new Date(r.timestamp).toLocaleString([], {
                          month: 'short',
                          day: '2-digit',
                          hour: '2-digit',
                          minute: '2-digit',
                          second: '2-digit',
                          hour12: false
                        }) : '—'}
                      </td>

                      {/* Verdict Badge */}
                      <td className="py-2.5 px-3">
                        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase
                          ${isPass 
                            ? 'bg-[#00f5a0]/15 text-[#00f5a0] border border-[#00f5a0]/30' 
                            : isReject 
                              ? 'bg-[#ff3b30]/15 text-[#ff817b] border border-[#ff3b30]/40' 
                              : 'bg-[#f59e0b]/15 text-[#f59e0b] border border-[#f59e0b]/30'}`
                        }>
                          {isPass && <ShieldCheck size={10} />}
                          {isReject && <ShieldAlert size={10} />}
                          {isUncertain && <AlertTriangle size={10} />}
                          {r.global_verdict}
                        </span>
                      </td>

                      {/* Primary Defect */}
                      <td className="py-2.5 px-3">
                        <span className={`font-semibold ${isReject ? 'text-[#ff817b]' : 'text-[#849588]'}`}>
                          {r.primary_defect ? r.primary_defect.replace(/_/g, ' ').toUpperCase() : 'NONE'}
                        </span>
                      </td>

                      {/* Confidence */}
                      <td className="py-2.5 px-3 text-[#dae2fd]">
                        {conf ? `${(conf * 100).toFixed(1)}%` : '—'}
                      </td>

                      {/* Flagged Cameras */}
                      <td className="py-2.5 px-3 text-[#849588]">
                        {r.flagged_cameras && r.flagged_cameras.length > 0 
                          ? r.flagged_cameras.map(c => c.replace('camera_', '')).join(', ').toUpperCase()
                          : 'ALL OPTIMAL'}
                      </td>

                      {/* Actuator Status */}
                      <td className="py-2.5 px-3">
                        {r.trigger_actuator ? (
                          <span className="inline-flex items-center gap-1 text-[#ff817b] font-bold">
                            <AlertOctagon size={12} /> FIRED
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-[#00f5a0]">
                            <CheckCircle2 size={12} /> STANDBY
                          </span>
                        )}
                      </td>

                      {/* Latency */}
                      <td className="py-2.5 px-3 text-[#00daf3]">
                        {r.total_latency_ms ? `${r.total_latency_ms.toFixed(1)} ms` : '—'}
                      </td>

                      {/* Actions */}
                      <td className="py-2.5 px-3 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectBottle && onSelectBottle(r.bottle_id);
                          }}
                          className="px-2.5 py-1 rounded bg-[#171f33] hover:bg-[#1f2c47] border border-[#1e2e48] hover:border-[#00daf3] text-[#00daf3] text-[11px] font-mono flex items-center gap-1 ml-auto transition-colors"
                        >
                          <Eye size={12} />
                          <span>REPLAY</span>
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Server-Side Pagination Bar */}
        <div className="p-3 bg-[#060e20] border-t border-[#1e2e48] flex flex-wrap items-center justify-between gap-3 text-xs font-mono text-[#849588]">
          <div className="flex items-center gap-3">
            <span>
              Showing {totalCount > 0 ? page * pageSize + 1 : 0} to {Math.min((page + 1) * pageSize, totalCount)} of {totalCount} records
            </span>
            <div className="flex items-center gap-1.5 ml-2">
              <span>ROWS:</span>
              <select
                value={pageSize}
                onChange={(e) => {
                  setPageSize(parseInt(e.target.value, 10));
                  setPage(0);
                }}
                className="bg-[#171f33] border border-[#1e2e48] text-xs text-[#dae2fd] rounded px-1.5 py-0.5 focus:outline-none"
              >
                <option value={10}>10</option>
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </div>
          </div>

          {/* Navigation Controls */}
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setPage(p => Math.max(p - 1, 0))}
              disabled={page === 0}
              className="p-1.5 rounded bg-[#171f33] border border-[#1e2e48] text-[#dae2fd] hover:bg-[#1f2c47] disabled:opacity-40 transition-colors"
              title="Previous Page"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="px-2 font-bold text-[#dae2fd]">
              PAGE {page + 1} OF {totalPages}
            </span>
            <button
              onClick={() => setPage(p => Math.min(p + 1, totalPages - 1))}
              disabled={page >= totalPages - 1}
              className="p-1.5 rounded bg-[#171f33] border border-[#1e2e48] text-[#dae2fd] hover:bg-[#1f2c47] disabled:opacity-40 transition-colors"
              title="Next Page"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
