import { useEffect, useCallback, useState } from 'react'
import {
  LayoutDashboard, Search, Activity, AlertOctagon,
  BarChart3, Settings, FileText,
} from 'lucide-react'

import { AppProvider, useApp } from './store'
import { useLiveInspections } from './hooks/useLiveInspections'
import {
  getAnalyticsSummary, getDefectPareto, getSeverityDistribution,
  getLatencyTrend, getHealth, downloadReport,
} from './api'

import { LoginPage } from './components/LoginPage'
import { TopNav } from './components/TopNav'
import { KPIRow } from './components/KPIRow'
import { LiveFeed } from './components/LiveFeed'
import { InspectPanel } from './components/InspectPanel'
import { InspectionTable } from './components/InspectionTable'
import { EscalationQueue } from './components/EscalationQueue'
import { DefectParetoChart, SeverityPieChart, LatencyTrendChart } from './components/Charts'
import { ModelInfoPanel } from './components/ModelInfoPanel'
import { DeviceStatusPanel } from './components/DeviceStatusPanel'

// Tab definitions with role access control
const TABS = [
  { id: 'dashboard',  label: 'Dashboard',  icon: LayoutDashboard },
  { id: 'live',       label: 'Live',        icon: BarChart3 },
  { id: 'inspect',    label: 'Inspect',     icon: Search },
  { id: 'history',    label: 'History',     icon: Activity },
  { id: 'escalation', label: 'Escalations', icon: AlertOctagon },
  { id: 'reports',    label: 'Reports',     icon: FileText, roles: ['supervisor', 'admin'] },
  { id: 'settings',   label: 'Settings',    icon: Settings, roles: ['admin'] },
] as const

type TabId = (typeof TABS)[number]['id']

// â”€â”€â”€ Inner app (needs AppProvider context) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function InnerApp() {
  const { state, dispatch } = useApp()
  const { auth, analytics } = state
  const [tab, setTab] = useState<TabId>('dashboard')
  const [latency, setLatency] = useState<import('./types').LatencyTrend[]>([])
  const [latencyLoading, setLatencyLoading] = useState(false)
  const [reportLoading, setReportLoading] = useState(false)

  // Boot: check API health
  useEffect(() => {
    getHealth()
      .then((h) => {
        dispatch({ type: 'SET_API_STATUS', payload: 'ok' })
        dispatch({ type: 'SET_MODEL_LOADED', payload: h.model_loaded })
      })
      .catch(() => dispatch({ type: 'SET_API_STATUS', payload: 'error' }))
  }, [dispatch])

  // Live WebSocket
  useLiveInspections()

  // Analytics loader
  const loadAnalytics = useCallback(async () => {
    dispatch({ type: 'SET_ANALYTICS_LOADING', payload: true })
    try {
      const [summary, pareto, severity] = await Promise.all([
        getAnalyticsSummary(analytics.hours),
        getDefectPareto(analytics.hours),
        getSeverityDistribution(analytics.hours),
      ])
      dispatch({ type: 'SET_ANALYTICS', payload: { summary, pareto, severity } })
    } catch {
      dispatch({ type: 'SET_ANALYTICS_LOADING', payload: false })
    }
  }, [dispatch, analytics.hours])

  const loadLatency = useCallback(async () => {
    setLatencyLoading(true)
    try {
      const data = await getLatencyTrend(analytics.hours)
      setLatency(data)
    } catch {
      setLatency([])
    } finally {
      setLatencyLoading(false)
    }
  }, [analytics.hours])

  useEffect(() => {
    if (tab === 'dashboard' || tab === 'live') {
      loadAnalytics()
      loadLatency()
    }
  }, [tab, loadAnalytics, loadLatency])

  // Auth gate
  if (!auth) return <LoginPage />

  async function handleDownloadReport(type: 'daily' | 'weekly') {
    setReportLoading(true)
    try {
      const blob = await downloadReport(type)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `visionfood-${type}-report.pdf`
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      // silent
    } finally {
      setReportLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-950 text-gray-100">
      <TopNav
        tab={tab}
        onTabChange={(t) => setTab(t as TabId)}
        tabs={TABS as unknown as { id: string; label: string; icon: React.ElementType; roles?: string[] }[]}
      />

      <main className="flex-1 w-full max-w-screen-2xl mx-auto px-3 md:px-6 py-4 md:py-6">

        {/* â”€â”€â”€â”€â”€â”€â”€â”€â”€ DASHBOARD â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        {tab === 'dashboard' && (
          <div className="space-y-5">
            {/* Time window selector + refresh */}
            <div className="flex items-center justify-between flex-wrap gap-2">
              <h2 className="text-base font-semibold text-gray-100">Overview</h2>
              <div className="flex items-center gap-2">
                <select
                  value={analytics.hours}
                  onChange={(e) => dispatch({ type: 'SET_ANALYTICS_HOURS', payload: Number(e.target.value) })}
                  className="bg-gray-800 text-gray-300 rounded-lg px-2 py-1.5 text-xs border border-gray-700 focus:outline-none"
                  aria-label="Time window"
                >
                  <option value={1}>Last 1h</option>
                  <option value={8}>Last 8h</option>
                  <option value={24}>Last 24h</option>
                  <option value={168}>Last 7d</option>
                </select>
                <button
                  onClick={() => { loadAnalytics(); loadLatency() }}
                  className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-xs transition-colors"
                  aria-label="Refresh"
                >
                  Refresh
                </button>
              </div>
            </div>

            <KPIRow summary={analytics.summary} loading={analytics.loading} />

            {/* Charts grid + live feed */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              <DefectParetoChart data={analytics.pareto} loading={analytics.loading} />
              <SeverityPieChart data={analytics.severity} loading={analytics.loading} />
              <div className="md:col-span-2 xl:col-span-1">
                <LiveFeed />
              </div>
            </div>

            <LatencyTrendChart data={latency} loading={latencyLoading} />
          </div>
        )}

        {/* â”€â”€â”€â”€â”€â”€â”€â”€â”€ LIVE STREAM â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        {tab === 'live' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <LiveFeed />
            <EscalationQueue />
          </div>
        )}

        {/* â”€â”€â”€â”€â”€â”€â”€â”€â”€ MANUAL INSPECT â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        {tab === 'inspect' && (
          <div className="max-w-xl mx-auto">
            <InspectPanel />
          </div>
        )}

        {/* â”€â”€â”€â”€â”€â”€â”€â”€â”€ HISTORY â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        {tab === 'history' && (
          <div>
            <h2 className="text-base font-semibold text-gray-100 mb-4">Inspection History</h2>
            <InspectionTable />
          </div>
        )}

        {/* â”€â”€â”€â”€â”€â”€â”€â”€â”€ ESCALATIONS â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        {tab === 'escalation' && (
          <div className="max-w-2xl mx-auto">
            <EscalationQueue />
          </div>
        )}

        {/* â”€â”€â”€â”€â”€â”€â”€â”€â”€ REPORTS (supervisor+) â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        {tab === 'reports' && (
          <div className="max-w-lg mx-auto space-y-4">
            <h2 className="text-base font-semibold text-gray-100">Reports</h2>
            <div className="bg-gray-900 rounded-xl p-5 space-y-3">
              <p className="text-gray-400 text-sm">Generate and download quality reports.</p>
              <div className="flex gap-3">
                <button
                  onClick={() => handleDownloadReport('daily')}
                  disabled={reportLoading}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white rounded-lg py-2.5 text-sm font-semibold transition-colors min-h-[44px]"
                >
                  {reportLoading ? 'Generatingâ€¦' : 'Daily Report'}
                </button>
                <button
                  onClick={() => handleDownloadReport('weekly')}
                  disabled={reportLoading}
                  className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 text-white rounded-lg py-2.5 text-sm font-semibold transition-colors min-h-[44px]"
                >
                  {reportLoading ? 'Generatingâ€¦' : 'Weekly Report'}
                </button>
              </div>
            </div>

            <div className="mt-6">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">Analytics Snapshot</h3>
              <KPIRow summary={analytics.summary} loading={analytics.loading} />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <DefectParetoChart data={analytics.pareto} loading={analytics.loading} />
                <SeverityPieChart data={analytics.severity} loading={analytics.loading} />
              </div>
            </div>
          </div>
        )}

        {/* â”€â”€â”€â”€â”€â”€â”€â”€â”€ SETTINGS (admin) â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        {tab === 'settings' && (
          <div className="max-w-2xl mx-auto space-y-4">
            <h2 className="text-base font-semibold text-gray-100">System Settings</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <ModelInfoPanel />
              <DeviceStatusPanel />
            </div>
            <div className="bg-gray-900 rounded-xl p-5 text-gray-500 text-sm text-center">
              Threshold controls and user management â€” fill in when backend endpoints are ready.
            </div>
          </div>
        )}
      </main>

      <footer className="border-t border-gray-800 py-2 text-center text-gray-700 text-xs">
        VisionFood QAI Â· Capstone 2026 Â· {auth.username} ({auth.role})
      </footer>
    </div>
  )
}

// â”€â”€â”€ Root: wrap with AppProvider â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export default function App() {
  return (
    <AppProvider>
      <InnerApp />
    </AppProvider>
  )
}
