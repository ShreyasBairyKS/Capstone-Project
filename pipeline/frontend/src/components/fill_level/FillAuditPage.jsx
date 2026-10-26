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
  Droplets,
  CheckCircle2, 
  AlertOctagon, 
  AlertTriangle 
} from 'lucide-react';

export default function FillAuditPage({ onSelectBottle }) {
  const [records, setRecords] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [copiedId, setCopiedId] = useState(null);

  // Filter States
  const [verdict, setVerdict] = useState('');
  const [fillStatus, setFillStatus] = useState('');
  const [searchTerm, setSearchTerm] = useState('');

  // Pagination States
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);

  const fillStatusOptions = [
    { id: '', label: 'ALL FILL STATES' },
    { id: 'normal', label: 'Nominal Volume (40-85%)' },
    { id: 'underfill', label: 'Underfill (<40%)' },
    { id: 'overfill', label: 'Overfill (>85%)' },
    { id: 'no_water', label: 'Dry / No Liquid' },
    { id: 'borderline', label: 'Borderline Gray-Zone' },
    { id: 'no_bottle', label: 'Conveyor Gap (No Bottle)' },
  ];

  const fetchRecords = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        limit: pageSize.toString(),
        offset: (page * pageSize).toString(),
        paginate: 'true'
      });
      if (verdict) params.set('verdict', verdict);
      if (fillStatus) params.set('fill_status', fillStatus);
      if (searchTerm.trim()) params.set('search', searchTerm.trim());

      const res = await fetch(`/api/fill/defects?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setRecords(data.items || []);
        setTotalCount(data.total_count || 0);
      }
    } catch (err) {
      console.error('Error fetching fill audit records:', err);
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, verdict, fillStatus, searchTerm]);

  useEffect(() => {
    fetchRecords();
  }, [fetchRecords]);

  // Copy Bottle ID
  const handleCopyId = (id, e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(id);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  // CSV Export
  const handleExportCsv = async () => {
    setExporting(true);
    try {
      const params = new URLSearchParams();
      if (verdict) params.set('verdict', verdict);
      if (fillStatus) params.set('fill_status', fillStatus);
      if (searchTerm.trim()) params.set('search', searchTerm.trim());

      const res = await fetch(`/api/fill/export-csv?${params.toString()}`);
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `fill_level_audit_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '_')}.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      }
    } catch (err) {
      console.error('Error exporting fill audit CSV:', err);
    } finally {
      setExporting(false);
    }
  };

  const totalPages = Math.ceil(totalCount / pageSize);

  return (
    <div className="flex flex-col gap-3.5 w-full">
      {/* Top Filter and Controls Bar */}
      <div className="p-3.5 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-wrap items-center justify-between gap-3 shadow-sm">
        {/* Left: Filter dropdowns */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Search input */}
          <div className="relative min-w-[200px]">
            <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[#849588]" />
            <input
              type="text"
              placeholder="Search Bottle ID..."
              value={searchTerm}
              onChange={e => {
                setSearchTerm(e.target.value);
                setPage(0);
              }}
              className="w-full pl-8 pr-3 py-1.5 rounded bg-[#060a14] border border-[#1e2e48] text-xs font-mono text-[#dae2fd] placeholder-[#849588] focus:outline-none focus:border-[#00daf3]"
            />
          </div>

          {/* Verdict Filter */}
          <select
            value={verdict}
            onChange={e => {
              setVerdict(e.target.value);
              setPage(0);
            }}
            className="px-2.5 py-1.5 rounded bg-[#060a14] border border-[#1e2e48] text-xs font-mono text-[#dae2fd] focus:outline-none focus:border-[#00daf3]"
          >
            <option value="">ALL VERDICTS</option>
            <option value="PASS">PASS ONLY</option>
            <option value="REJECT">REJECT ONLY</option>
            <option value="UNCERTAIN">UNCERTAIN ONLY</option>
          </select>

          {/* Fill Status Filter */}
          <select
            value={fillStatus}
            onChange={e => {
              setFillStatus(e.target.value);
              setPage(0);
            }}
            className="px-2.5 py-1.5 rounded bg-[#060a14] border border-[#1e2e48] text-xs font-mono text-[#dae2fd] focus:outline-none focus:border-[#00daf3]"
          >
            {fillStatusOptions.map(opt => (
              <option key={opt.id} value={opt.id}>{opt.label}</option>
            ))}
          </select>

          {/* Reset Filters */}
          {(verdict || fillStatus || searchTerm) && (
            <button
              onClick={() => {
                setVerdict('');
                setFillStatus('');
                setSearchTerm('');
                setPage(0);
              }}
              className="text-[11px] font-mono text-[#849588] hover:text-[#00daf3] underline"
            >
              Reset Filters
            </button>
          )}
        </div>

        {/* Right: Refresh & CSV Export */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => fetchRecords()}
            disabled={loading}
            className="p-1.5 rounded bg-[#17253d] border border-[#1e2e48] text-[#849588] hover:text-[#dae2fd] hover:border-[#00daf3] transition-colors"
            title="Refresh Table"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin text-[#00daf3]' : ''} />
          </button>

          <button
            onClick={handleExportCsv}
            disabled={exporting}
            className="px-3 py-1.5 rounded bg-[#17253d] border border-[#00daf3]/40 text-[#00daf3] hover:bg-[#1f3354] transition-colors text-xs font-mono font-bold flex items-center gap-1.5"
          >
            <Download size={13} />
            <span>{exporting ? 'EXPORTING CSV...' : 'EXPORT RFC CSV'}</span>
          </button>
        </div>
      </div>

      {/* Main Audit Telemetry Table */}
      <div className="rounded bg-[#0b1326] border border-[#1e2e48] overflow-hidden shadow-md">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="bg-[#070e1e] border-b border-[#1e2e48] text-[#849588] text-[10px] font-bold tracking-wider uppercase select-none">
                <th className="p-3">SPECIMEN ID</th>
                <th className="p-3">TIMESTAMP</th>
                <th className="p-3">VERDICT</th>
                <th className="p-3">FILL STATUS</th>
                <th className="p-3">FILL RATIO</th>
                <th className="p-3">WATER Y</th>
                <th className="p-3">BOTTLE H</th>
                <th className="p-3">ACTUATOR</th>
                <th className="p-3">LATENCY</th>
                <th className="p-3 text-right">ACTION</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1e2e48]/50 text-[#dae2fd]">
              {records.length === 0 ? (
                <tr>
                  <td colSpan={10} className="p-8 text-center text-[#849588]">
                    {loading ? 'Querying Station 2 SQLite audit database...' : 'No fill inspection records found matching active filters.'}
                  </td>
                </tr>
              ) : (
                records.map((r) => {
                  const isPass = r.global_verdict === 'PASS';
                  const isReject = r.global_verdict === 'REJECT';
                  const isUnc = r.global_verdict === 'UNCERTAIN';
                  const ratio = r.fill_ratio != null ? (r.fill_ratio * 100).toFixed(1) : '--';

                  return (
                    <tr 
                      key={r.id || r.bottle_id}
                      className="hover:bg-[#131b2e]/60 transition-colors cursor-pointer"
                      onClick={() => onSelectBottle?.(r.bottle_id)}
                    >
                      {/* Bottle ID with Copy */}
                      <td className="p-3 font-bold text-[#00daf3] flex items-center gap-1.5">
                        <span title={r.bottle_id}>{r.bottle_id}</span>
                        <button
                          onClick={(e) => handleCopyId(r.bottle_id, e)}
                          className="text-[#849588] hover:text-[#dae2fd] p-0.5 rounded"
                          title="Copy Bottle ID"
                        >
                          {copiedId === r.bottle_id ? <Check size={11} className="text-[#00f5a0]" /> : <Copy size={11} />}
                        </button>
                      </td>

                      {/* Timestamp */}
                      <td className="p-3 text-[#849588] whitespace-nowrap">
                        {r.timestamp ? r.timestamp.slice(11, 19) : '--'}
                      </td>

                      {/* Verdict */}
                      <td className="p-3">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold border inline-flex items-center gap-1
                          ${isPass ? 'bg-[#00f5a0]/10 text-[#00f5a0] border-[#00f5a0]/30' : ''}
                          ${isReject ? 'bg-[#ff3b30]/10 text-[#ff817b] border-[#ff3b30]/30' : ''}
                          ${isUnc ? 'bg-[#f59e0b]/10 text-[#fbbf24] border-[#f59e0b]/30' : ''}`
                        }>
                          {isPass && <CheckCircle2 size={10} />}
                          {isReject && <AlertOctagon size={10} />}
                          {isUnc && <AlertTriangle size={10} />}
                          <span>{r.global_verdict}</span>
                        </span>
                      </td>

                      {/* Fill Status */}
                      <td className="p-3 uppercase font-semibold text-[#849588]">
                        <span className={isReject ? 'text-[#ff817b]' : ''}>{r.fill_status}</span>
                      </td>

                      {/* Fill Ratio with mini bar */}
                      <td className="p-3">
                        <div className="flex items-center gap-2">
                          <span className="font-bold w-12">{ratio}%</span>
                          <div className="w-16 h-1.5 bg-[#171f33] rounded overflow-hidden">
                            <div 
                              className={`h-full ${isPass ? 'bg-[#00f5a0]' : isReject ? 'bg-[#ff3b30]' : 'bg-[#fbbf24]'}`}
                              style={{ width: `${Math.min(100, Math.max(0, r.fill_ratio ? r.fill_ratio * 100 : 0))}%` }}
                            />
                          </div>
                        </div>
                      </td>

                      {/* Water Y */}
                      <td className="p-3 text-[#849588]">{r.water_y != null ? `${r.water_y} px` : '--'}</td>

                      {/* Bottle H */}
                      <td className="p-3 text-[#849588]">{r.bottle_h != null ? `${r.bottle_h} px` : '--'}</td>

                      {/* Actuator */}
                      <td className="p-3">
                        <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold
                          ${r.trigger_actuator 
                            ? 'bg-[#ff3b30]/20 text-[#ff817b] border border-[#ff3b30]/40' 
                            : 'bg-[#171f33] text-[#849588]'}`
                        }>
                          {r.trigger_actuator ? 'EJECT FIRED' : 'ARMED'}
                        </span>
                      </td>

                      {/* Latency */}
                      <td className="p-3 text-[#00f5a0]">
                        {r.total_latency_ms != null ? `${r.total_latency_ms.toFixed(1)} ms` : '--'}
                      </td>

                      {/* Action */}
                      <td className="p-3 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectBottle?.(r.bottle_id);
                          }}
                          className="px-2 py-1 rounded bg-[#17253d] border border-[#1e2e48] hover:border-[#00daf3] text-[#00daf3] text-[10px] font-bold inline-flex items-center gap-1 transition-colors"
                        >
                          <Eye size={11} />
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

        {/* Pagination Footer */}
        <div className="p-3 border-t border-[#1e2e48] bg-[#070e1e] flex flex-wrap items-center justify-between gap-2 text-xs font-mono text-[#849588]">
          <div>
            SHOWING {records.length > 0 ? page * pageSize + 1 : 0} TO {Math.min((page + 1) * pageSize, totalCount)} OF {totalCount} SPECIMENS
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(prev => Math.max(0, prev - 1))}
              disabled={page === 0}
              className="p-1 rounded bg-[#171f33] border border-[#1e2e48] disabled:opacity-40 disabled:cursor-not-allowed hover:text-[#dae2fd] text-[#849588]"
            >
              <ChevronLeft size={16} />
            </button>

            <span>PAGE {totalPages > 0 ? page + 1 : 1} OF {totalPages || 1}</span>

            <button
              onClick={() => setPage(prev => Math.min(totalPages - 1, prev + 1))}
              disabled={page >= totalPages - 1}
              className="p-1 rounded bg-[#171f33] border border-[#1e2e48] disabled:opacity-40 disabled:cursor-not-allowed hover:text-[#dae2fd] text-[#849588]"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
