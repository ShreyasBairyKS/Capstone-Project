import { memo } from 'react'
import type { AnalyticsSummary } from '../types'
import { TrendingDown, CheckCircle, Clock, BarChart2, AlertOctagon, XCircle } from 'lucide-react'

interface Props {
  summary: AnalyticsSummary | null
  loading: boolean
}

function KPICard({
  label, value, sub, icon: Icon, colour, highlight,
}: {
  label: string; value: string; sub?: string
  icon: React.ElementType; colour: string
  highlight?: boolean
}) {
  return (
    <div className={`card-sm flex items-start gap-3 ${highlight ? 'border-l-2' : ''}`}
      style={highlight ? { borderLeftColor: 'var(--highlight)' } : undefined}>
      <div className={`w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 ${colour}`}>
        <Icon size={16} aria-hidden />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-gray-500 text-[11px] uppercase tracking-widest font-semibold mb-1 truncate">{label}</p>
        <p className="text-xl font-bold text-white leading-none">{value}</p>
        {sub && <p className="text-gray-500 text-xs mt-1 truncate">{sub}</p>}
      </div>
    </div>
  )
}

export const KPIRow = memo(function KPIRow({ summary, loading }: Props) {
  if (loading || !summary) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="card-sm h-20 animate-pulse" />
        ))}
      </div>
    )
  }

  const failCount = summary.by_verdict['FAIL'] ?? 0
  const escalateCount = summary.by_verdict['ESCALATE'] ?? 0
  const passCount = summary.by_verdict['PASS'] ?? 0
  const passPct = summary.total_inspections > 0
    ? ((passCount / summary.total_inspections) * 100).toFixed(1)
    : '0.0'
  const defectPct = (summary.defect_rate * 100).toFixed(1)

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
      <KPICard
        label="Total" value={summary.total_inspections.toLocaleString()}
        sub={`Last ${summary.window_hours}h`}
        icon={BarChart2} colour="bg-blue-500/15 text-blue-400" />
      <KPICard
        label="Pass Rate" value={`${passPct}%`}
        sub={`${passCount.toLocaleString()} passed`}
        icon={CheckCircle} colour="bg-brand-500/15 text-brand-400" />
      <KPICard
        label="Defect Rate" value={`${defectPct}%`}
        sub="of all inspections"
        icon={TrendingDown} colour="bg-red-500/15 text-red-400" />
      <KPICard
        label="FAIL" value={failCount.toLocaleString()}
        sub="Hard failures"
        icon={XCircle} colour="bg-red-500/15 text-red-400" />
      <KPICard
        label="Escalated" value={escalateCount.toLocaleString()}
        sub="Needs review"
        icon={AlertOctagon} colour="bg-orange-500/15 text-orange-400" />
      <KPICard
        label="Avg Latency" value={`${summary.avg_latency_ms.toFixed(0)}ms`}
        sub="Edge inference"
        icon={Clock} colour="bg-purple-500/15 text-purple-400" />
    </div>
  )
})
