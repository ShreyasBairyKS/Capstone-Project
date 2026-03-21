import type { AnalyticsSummary } from '../types'
import { TrendingDown, CheckCircle, AlertTriangle, Clock } from 'lucide-react'

interface Props {
  summary: AnalyticsSummary | null
  loading: boolean
}

function KPICard({
  label,
  value,
  sub,
  icon: Icon,
  colour,
}: {
  label: string
  value: string
  sub?: string
  icon: React.ElementType
  colour: string
}) {
  return (
    <div className="bg-gray-900 rounded-xl p-5 flex items-start gap-4">
      <div className={`p-2 rounded-lg ${colour}`}>
        <Icon size={22} />
      </div>
      <div>
        <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">{label}</p>
        <p className="text-2xl font-bold text-white">{value}</p>
        {sub && <p className="text-gray-500 text-xs mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}

export function KPIRow({ summary, loading }: Props) {
  if (loading || !summary) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-gray-900 rounded-xl p-5 h-24 animate-pulse" />
        ))}
      </div>
    )
  }

  const defectPct = (summary.defect_rate * 100).toFixed(1)
  const failCount = summary.by_verdict['FAIL'] ?? 0
  const escalateCount = summary.by_verdict['ESCALATE'] ?? 0
  const passCount = summary.by_verdict['PASS'] ?? 0
  const passPct = summary.total_inspections > 0
    ? ((passCount / summary.total_inspections) * 100).toFixed(1)
    : '0.0'

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <KPICard
        label="Total Inspections"
        value={summary.total_inspections.toLocaleString()}
        sub={`Last ${summary.window_hours}h`}
        icon={CheckCircle}
        colour="bg-blue-500/20 text-blue-400"
      />
      <KPICard
        label="Pass Rate"
        value={`${passPct}%`}
        sub={`${passCount} passed`}
        icon={CheckCircle}
        colour="bg-green-500/20 text-green-400"
      />
      <KPICard
        label="Defect Rate"
        value={`${defectPct}%`}
        sub={`${failCount} FAIL  ${escalateCount} ESCALATE`}
        icon={TrendingDown}
        colour="bg-red-500/20 text-red-400"
      />
      <KPICard
        label="Avg Latency"
        value={`${summary.avg_latency_ms.toFixed(0)} ms`}
        sub="Edge inference"
        icon={Clock}
        colour="bg-purple-500/20 text-purple-400"
      />
    </div>
  )
}
