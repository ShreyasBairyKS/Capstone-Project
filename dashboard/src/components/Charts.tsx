import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  Legend,
} from 'recharts'
import type { DefectPareto, SeverityDistribution } from '../types'

const CLASS_COLOURS: Record<string, string> = {
  improper_filling:    '#3b82f6',
  packaging_damage:    '#f97316',
  label_misalignment:  '#a855f7',
  surface_contamination: '#ef4444',
}

const SEVERITY_COLOURS: Record<string, string> = {
  S1: '#22c55e',
  S2: '#eab308',
  S3: '#f97316',
  S4: '#ef4444',
}

// ------------------------------------------------------------------ //
// Pareto Bar chart
// ------------------------------------------------------------------ //

interface ParetoProps {
  data: DefectPareto[]
  loading: boolean
}

export function DefectParetoChart({ data, loading }: ParetoProps) {
  return (
    <div className="bg-gray-900 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-gray-300 mb-4 uppercase tracking-wider">
        Defect Class Breakdown
      </h3>
      {loading ? (
        <div className="h-48 animate-pulse bg-gray-800 rounded-lg" />
      ) : data.length === 0 ? (
        <p className="text-gray-500 text-sm py-12 text-center">No defects recorded</p>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
            <XAxis
              dataKey="class_name"
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              tickFormatter={(v) => v.replace('_', ' ')}
            />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
              labelStyle={{ color: '#e5e7eb' }}
              formatter={(value: number, _name: string, props) => [`${value} (${props.payload.pct?.toFixed(1)}%)`, 'Count']}
            />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {data.map((entry) => (
                <Cell key={entry.class_name} fill={CLASS_COLOURS[entry.class_name] ?? '#6b7280'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ //
// Severity Pie chart
// ------------------------------------------------------------------ //

interface SeverityProps {
  data: SeverityDistribution[]
  loading: boolean
}

export function SeverityPieChart({ data, loading }: SeverityProps) {
  const pieData = data.map((d) => ({ name: d.grade, value: d.count }))
  return (
    <div className="bg-gray-900 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-gray-300 mb-4 uppercase tracking-wider">
        Severity Distribution
      </h3>
      {loading ? (
        <div className="h-48 animate-pulse bg-gray-800 rounded-lg" />
      ) : pieData.length === 0 ? (
        <p className="text-gray-500 text-sm py-12 text-center">No severity data</p>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={80}
              paddingAngle={3}
              dataKey="value"
            >
              {pieData.map((entry) => (
                <Cell key={entry.name} fill={SEVERITY_COLOURS[entry.name] ?? '#6b7280'} />
              ))}
            </Pie>
            <Legend
              formatter={(value) => <span style={{ color: '#9ca3af', fontSize: 12 }}>{value}</span>}
            />
            <Tooltip
              contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
              labelStyle={{ color: '#e5e7eb' }}
            />
          </PieChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
