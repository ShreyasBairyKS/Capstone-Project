import { useEffect, useState } from 'react'
import { listInspections } from '../api'
import type { InspectionSummary, Verdict } from '../types'
import { VerdictBadge } from './VerdictBadge'
import { RefreshCw } from 'lucide-react'

export function InspectionTable() {
  const [rows, setRows] = useState<InspectionSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<Verdict | ''>('')

  async function load() {
    setLoading(true)
    try {
      const data = await listInspections(50, 0, filter || undefined)
      setRows(data)
    } catch {
      // silent — could show error state
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [filter])

  return (
    <div className="bg-gray-900 rounded-xl p-5">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          Recent Inspections
        </h3>
        <div className="flex items-center gap-2">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as Verdict | '')}
            className="bg-gray-800 text-gray-300 rounded-lg px-3 py-1.5 text-xs border border-gray-700 focus:outline-none"
          >
            <option value="">All verdicts</option>
            <option value="PASS">PASS</option>
            <option value="FAIL">FAIL</option>
            <option value="ESCALATE">ESCALATE</option>
            <option value="REVIEW">REVIEW</option>
          </select>
          <button
            onClick={load}
            className="p-1.5 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-400 transition-colors"
            title="Refresh"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-gray-500 text-xs border-b border-gray-800">
              <th className="pb-2 pr-4">Product</th>
              <th className="pb-2 pr-4">SKU</th>
              <th className="pb-2 pr-4">Verdict</th>
              <th className="pb-2 pr-4">Defects</th>
              <th className="pb-2 pr-4">Latency</th>
              <th className="pb-2">Time</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              [...Array(6)].map((_, i) => (
                <tr key={i}>
                  {[...Array(6)].map((_, j) => (
                    <td key={j} className="py-2 pr-4">
                      <div className="h-4 bg-gray-800 rounded animate-pulse w-20" />
                    </td>
                  ))}
                </tr>
              ))
            ) : rows.length === 0 ? (
              <tr>
                <td colSpan={6} className="py-12 text-center text-gray-500">
                  No inspections found
                </td>
              </tr>
            ) : (
              rows.map((row) => (
                <tr
                  key={row.id}
                  className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors"
                >
                  <td className="py-2 pr-4 font-mono text-xs text-gray-400">
                    {row.product_id ?? '—'}
                  </td>
                  <td className="py-2 pr-4 text-gray-300">{row.sku}</td>
                  <td className="py-2 pr-4">
                    <VerdictBadge verdict={row.verdict} size="sm" />
                  </td>
                  <td className="py-2 pr-4 text-gray-400">{row.defect_count}</td>
                  <td className="py-2 pr-4 text-gray-400 font-mono text-xs">
                    {row.latency_ms.toFixed(0)} ms
                  </td>
                  <td className="py-2 text-gray-500 text-xs">
                    {new Date(row.timestamp).toLocaleTimeString()}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
