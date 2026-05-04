import { useState } from 'react'
import { FileText, Download, Calendar, BarChart3 } from 'lucide-react'
import { downloadReport } from '../api'
import { KPIRow } from './KPIRow'
import { DefectParetoChart, SeverityPieChart, LatencyTrendChart } from './Charts'
import { useToast } from '../store'
import type { AnalyticsSummary, DefectPareto, SeverityDistribution, LatencyTrend } from '../types'

interface AnalyticsState {
  summary: AnalyticsSummary | null
  pareto: DefectPareto[]
  severity: SeverityDistribution[]
  loading: boolean
  hours: number
}

interface Props {
  analytics: AnalyticsState
  latency: LatencyTrend[]
  latencyLoading: boolean
}

type ReportType = 'daily' | 'weekly'
type ReportFormat = 'pdf'

export function ReportsPage({ analytics, latency, latencyLoading }: Props) {
  const toast = useToast()
  const [reportType, setReportType] = useState<ReportType>('daily')
  const [loading, setLoading] = useState(false)

  async function handleGenerate() {
    setLoading(true)
    try {
      const blob = await downloadReport(reportType)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `visionfood-${reportType}-report.pdf`
      a.click()
      URL.revokeObjectURL(url)
      toast('success', 'Report downloaded', `${reportType} report saved to your downloads folder`)
    } catch (e) {
      toast('error', 'Report generation failed', e instanceof Error ? e.message : 'Please try again')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      {/* Page header */}
      <div>
        <h1 className="page-title">Reports</h1>
        <p className="page-sub">Generate and download quality inspection reports</p>
      </div>

      {/* Generator card */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-1 card space-y-5">
          <div className="card-header">
            <FileText size={15} className="text-brand-400" />
            <span>Generate Report</span>
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-xs text-gray-400 block mb-1.5">Report Period</label>
              <div className="grid grid-cols-2 gap-2">
                {(['daily', 'weekly'] as ReportType[]).map((t) => (
                  <button
                    key={t}
                    onClick={() => setReportType(t)}
                    className={`py-2.5 rounded-xl text-sm font-medium border transition-all ${
                      reportType === t
                        ? 'bg-brand-500/20 border-brand-500/60 text-brand-300'
                        : 'bg-gray-800/60 border-gray-700 text-gray-400 hover:border-gray-600'
                    }`}
                  >
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="text-xs text-gray-400 block mb-1.5">Format</label>
              <div className="flex items-center gap-2 px-3 py-2.5 bg-gray-800/60 border border-gray-700 rounded-xl text-sm text-gray-300">
                <FileText size={13} className="text-red-400" />
                PDF Report
              </div>
            </div>

            <div className="pt-1">
              <p className="text-xs text-gray-500 mb-3 flex items-start gap-1.5">
                <Calendar size={11} className="mt-0.5 flex-shrink-0" />
                {reportType === 'daily'
                  ? 'Generates a comprehensive daily quality summary for the last 24 hours.'
                  : 'Generates a weekly quality trend report covering the last 7 days.'
                }
              </p>
              <button
                onClick={handleGenerate}
                disabled={loading}
                className="btn btn-primary w-full"
              >
                {loading ? (
                  <span className="flex items-center gap-2 justify-center">
                    <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Generating...
                  </span>
                ) : (
                  <span className="flex items-center gap-2 justify-center">
                    <Download size={14} />
                    Download {reportType.charAt(0).toUpperCase() + reportType.slice(1)} Report
                  </span>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Quick stats preview */}
        <div className="lg:col-span-2 space-y-5">
          <div className="card">
            <div className="card-header mb-4">
              <BarChart3 size={15} className="text-brand-400" />
              <span>Analytics Preview</span>
            </div>
            <KPIRow summary={analytics.summary} loading={analytics.loading} />
          </div>
        </div>
      </div>

      {/* Charts */}
      <div>
        <h2 className="text-sm font-semibold text-gray-300 mb-4">Data Visualisation</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <DefectParetoChart data={analytics.pareto} loading={analytics.loading} />
          <SeverityPieChart data={analytics.severity} loading={analytics.loading} />
        </div>
        <div className="mt-5">
          <LatencyTrendChart data={latency} loading={latencyLoading} />
        </div>
      </div>

      {/* Empty history placeholder */}
      <div className="card">
        <div className="card-header mb-4">
          <FileText size={15} className="text-gray-500" />
          <span>Report History</span>
        </div>
        <div className="flex flex-col items-center justify-center py-10 text-gray-600 gap-2">
          <FileText size={32} />
          <p className="text-sm">No previously generated reports</p>
          <p className="text-xs">Generated reports will appear here</p>
        </div>
      </div>
    </div>
  )
}
