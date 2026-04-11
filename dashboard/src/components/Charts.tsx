import { memo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend,
  LineChart, Line, CartesianGrid,
} from 'recharts'
import type { DefectPareto, SeverityDistribution, LatencyTrend } from '../types'

const CLASS_COLOURS: Record<string, string> = {
  improper_filling:     '#3b82f6',
  packaging_damage:     '#f97316',
  label_misalignment:   '#a855f7',
  surface_contamination:'#ef4444',
}

const SEVERITY_COLOURS: Record<string, string> = {
  S1: '#22c55e', S2: '#eab308', S3: '#f97316', S4: '#ef4444',
}

const TOOLTIP_STYLE = {
  contentStyle: { background: '#111827', border: '1px solid #374151', borderRadius: 8 },
  labelStyle: { color: '#e5e7eb' },
}

// â”€â”€â”€ Skeleton â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function ChartSkeleton() {
  return <div className="h-48 animate-pulse bg-gray-800 rounded-lg" />
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-gray-900 rounded-xl p-4 md:p-5">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">{title}</h3>
      {children}
    </div>
  )
}

// â”€â”€â”€ Defect Pareto â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export const DefectParetoChart = memo(function DefectParetoChart({
  data, loading,
}: { data: DefectPareto[]; loading: boolean }) {
  return (
    <ChartCard title="Defect Class Breakdown">
      {loading ? <ChartSkeleton /> : data.length === 0 ? (
        <p className="text-gray-500 text-sm py-12 text-center">No defects recorded</p>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
            <XAxis dataKey="class_name" tick={{ fill: '#9ca3af', fontSize: 10 }}
              tickFormatter={(v) => v.replace(/_/g, ' ').replace('surface contamination', 'contam.')} />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} />
            <Tooltip {...TOOLTIP_STYLE}
              formatter={(v: number, _n: string, p) => [`${v} (${p.payload.pct?.toFixed(1)}%)`, 'Count']} />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {data.map((e) => (
                <Cell key={e.class_name} fill={CLASS_COLOURS[e.class_name] ?? '#6b7280'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </ChartCard>
  )
})

// â”€â”€â”€ Severity Pie â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export const SeverityPieChart = memo(function SeverityPieChart({
  data, loading,
}: { data: SeverityDistribution[]; loading: boolean }) {
  const pieData = data.map((d) => ({ name: d.grade, value: d.count }))
  return (
    <ChartCard title="Severity Distribution">
      {loading ? <ChartSkeleton /> : pieData.length === 0 ? (
        <p className="text-gray-500 text-sm py-12 text-center">No severity data</p>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={pieData} cx="50%" cy="50%" innerRadius={45} outerRadius={75} paddingAngle={3} dataKey="value">
              {pieData.map((e) => (
                <Cell key={e.name} fill={SEVERITY_COLOURS[e.name] ?? '#6b7280'} />
              ))}
            </Pie>
            <Legend formatter={(v) => <span style={{ color: '#9ca3af', fontSize: 11 }}>{v}</span>} />
            <Tooltip {...TOOLTIP_STYLE} />
          </PieChart>
        </ResponsiveContainer>
      )}
    </ChartCard>
  )
})

// â”€â”€â”€ Latency Trend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export const LatencyTrendChart = memo(function LatencyTrendChart({
  data, loading,
}: { data: LatencyTrend[]; loading: boolean }) {
  return (
    <ChartCard title="Inference Latency Trend">
      {loading ? <ChartSkeleton /> : data.length === 0 ? (
        <p className="text-gray-500 text-sm py-12 text-center">No latency data</p>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="timestamp" tick={{ fill: '#9ca3af', fontSize: 10 }}
              tickFormatter={(v) => new Date(v).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} unit=" ms" />
            <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(0)} ms`]} />
            <Line type="monotone" dataKey="p50_ms" stroke="#3b82f6" dot={false} name="p50" strokeWidth={2} />
            <Line type="monotone" dataKey="p95_ms" stroke="#f97316" dot={false} name="p95" strokeWidth={2} />
            <Line type="monotone" dataKey="p99_ms" stroke="#ef4444" dot={false} name="p99" strokeWidth={1} strokeDasharray="4 2" />
            <Legend formatter={(v) => <span style={{ color: '#9ca3af', fontSize: 11 }}>{v}</span>} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </ChartCard>
  )
})
