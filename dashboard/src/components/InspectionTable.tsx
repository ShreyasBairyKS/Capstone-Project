import { useEffect, useCallback, memo } from 'react'
import { listInspections, exportInspections } from '../api'
import { useApp } from '../store'
import type { Verdict } from '../types'
import { VerdictBadge } from './VerdictBadge'
import { RefreshCw, Download, Filter } from 'lucide-react'

export const InspectionTable = memo(function InspectionTable({ onRowClick }: { onRowClick?: (id: string) => void }) {
  const { state, dispatch } = useApp()
  const { rows, loading, filters } = state.history

  const load = useCallback(async () => {
    dispatch({ type: 'SET_HISTORY_LOADING', payload: true })
    try {
      const data = await listInspections({
        limit: 50,
        verdict: filters.verdict || undefined,
        sku: filters.sku || undefined,
        device_id: filters.deviceId || undefined,
        date_from: filters.dateFrom || undefined,
        date_to: filters.dateTo || undefined,
        escalated_only: filters.escalatedOnly || undefined,
      })
      dispatch({ type: 'SET_HISTORY', payload: data })
    } catch {
      dispatch({ type: 'SET_HISTORY_LOADING', payload: false })
    }
  }, [dispatch, filters])

  useEffect(() => { load() }, [load])

  async function handleExport(fmt: 'csv' | 'json') {
    try {
      const blob = await exportInspections(fmt)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `inspections.${fmt}`
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      // silent â€” could show toast in a real app
    }
  }

  return (
    <div className="bg-gray-900 rounded-xl p-4 md:p-5 space-y-4">
      {/* Filters row */}
      <div className="flex flex-wrap gap-2 items-end">
        <div className="flex items-center gap-1 text-gray-400 text-xs uppercase tracking-wider">
          <Filter size={12} aria-hidden />Filters
        </div>

        {/* Verdict */}
        <select
          value={filters.verdict}
          onChange={(e) => dispatch({ type: 'SET_HISTORY_FILTERS', payload: { verdict: e.target.value } })}
          className="bg-gray-800 text-gray-300 rounded-lg px-2 py-1.5 text-xs border border-gray-700 focus:outline-none focus:border-blue-500"
          aria-label="Filter by verdict"
        >
          <option value="">All verdicts</option>
          {(['PASS', 'FAIL', 'ESCALATE', 'REVIEW'] as Verdict[]).map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>

        {/* SKU */}
        <input
          type="text"
          value={filters.sku}
          onChange={(e) => dispatch({ type: 'SET_HISTORY_FILTERS', payload: { sku: e.target.value } })}
          placeholder="SKUâ€¦"
          className="bg-gray-800 text-gray-300 rounded-lg px-2 py-1.5 text-xs border border-gray-700 focus:outline-none focus:border-blue-500 w-28"
          aria-label="Filter by SKU"
        />

        {/* Device */}
        <input
          type="text"
          value={filters.deviceId}
          onChange={(e) => dispatch({ type: 'SET_HISTORY_FILTERS', payload: { deviceId: e.target.value } })}
          placeholder="Device IDâ€¦"
          className="bg-gray-800 text-gray-300 rounded-lg px-2 py-1.5 text-xs border border-gray-700 focus:outline-none focus:border-blue-500 w-28"
          aria-label="Filter by device"
        />

        {/* Date from */}
        <input
          type="date"
          value={filters.dateFrom}
          onChange={(e) => dispatch({ type: 'SET_HISTORY_FILTERS', payload: { dateFrom: e.target.value } })}
          className="bg-gray-800 text-gray-300 rounded-lg px-2 py-1.5 text-xs border border-gray-700 focus:outline-none focus:border-blue-500"
          aria-label="Date from"
        />

        {/* Date to */}
        <input
          type="date"
          value={filters.dateTo}
          onChange={(e) => dispatch({ type: 'SET_HISTORY_FILTERS', payload: { dateTo: e.target.value } })}
          className="bg-gray-800 text-gray-300 rounded-lg px-2 py-1.5 text-xs border border-gray-700 focus:outline-none focus:border-blue-500"
          aria-label="Date to"
        />

        {/* Escalated only */}
        <label className="flex items-center gap-1.5 text-xs text-gray-400 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={filters.escalatedOnly}
            onChange={(e) => dispatch({ type: 'SET_HISTORY_FILTERS', payload: { escalatedOnly: e.target.checked } })}
            className="accent-orange-500"
          />
          Escalated only
        </label>

        {/* Actions */}
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => handleExport('csv')}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-xs transition-colors"
            title="Export CSV"
          >
            <Download size={12} aria-hidden />CSV
          </button>
          <button
            onClick={() => handleExport('json')}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-xs transition-colors"
            title="Export JSON"
          >
            <Download size={12} aria-hidden />JSON
          </button>
          <button
            onClick={load}
            className="p-1.5 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-400 transition-colors"
            title="Refresh"
            aria-label="Refresh table"
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} aria-hidden />
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto -mx-1">
        <table className="w-full text-sm min-w-[600px]">
          <thead>
            <tr className="text-left text-gray-500 text-xs border-b border-gray-800">
              <th className="pb-2 pr-4 font-medium">Product</th>
              <th className="pb-2 pr-4 font-medium">SKU</th>
              <th className="pb-2 pr-4 font-medium">Verdict</th>
              <th className="pb-2 pr-4 font-medium">Defects</th>
              <th className="pb-2 pr-4 font-medium">Latency</th>
              <th className="pb-2 pr-4 font-medium">Device</th>
              <th className="pb-2 font-medium">Time</th>
            </tr>
          </thead>
          <tbody>
            {loading
              ? [...Array(6)].map((_, i) => (
                <tr key={i}>
                  {[...Array(7)].map((_, j) => (
                    <td key={j} className="py-2 pr-4">
                      <div className="h-4 bg-gray-800 rounded animate-pulse w-16" />
                    </td>
                  ))}
                </tr>
              ))
              : rows.length === 0
              ? (
                <tr>
                  <td colSpan={7} className="py-12 text-center text-gray-500">
                    No inspections match the current filters
                  </td>
                </tr>
              )
              : rows.map((row) => (
                <tr
                  key={row.id}
                  onClick={onRowClick ? () => onRowClick(row.id) : undefined}
                  className={`border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors ${
                    onRowClick ? 'cursor-pointer' : ''
                  }`}
                >
                  <td className="py-2 pr-4 font-mono text-xs text-gray-400 max-w-[100px] truncate">
                    {row.product_id ?? '-'}
                  </td>
                  <td className="py-2 pr-4 text-gray-300 text-xs">{row.sku}</td>
                  <td className="py-2 pr-4">
                    <VerdictBadge verdict={row.verdict} size="sm" />
                    {row.escalated && (
                      <span className="ml-1 text-orange-400 text-xs">â†‘</span>
                    )}
                  </td>
                  <td className="py-2 pr-4 text-gray-400 text-xs">{row.defect_count}</td>
                  <td className="py-2 pr-4 text-gray-400 font-mono text-xs">{row.latency_ms.toFixed(0)} ms</td>
                  <td className="py-2 pr-4 text-gray-400 text-xs font-mono truncate max-w-[80px]">{row.device_id}</td>
                  <td className="py-2 text-gray-500 text-xs whitespace-nowrap">
                    {new Date(row.timestamp).toLocaleTimeString()}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  )
})
