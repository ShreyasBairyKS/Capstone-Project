import { useEffect, useCallback, useState } from 'react'
import { RefreshCw } from 'lucide-react'

import { AppProvider, useApp } from './store'
import { useLiveInspections } from './hooks/useLiveInspections'
import {
  getAnalyticsSummary, getDefectPareto, getSeverityDistribution,
  getLatencyTrend, getHealth,
} from './api'

import { LoginPage }     from './components/LoginPage'
import { Sidebar }       from './components/Sidebar'
import { KPIRow }        from './components/KPIRow'
import { LiveFeed }      from './components/LiveFeed'
import { InspectPanel }  from './components/InspectPanel'
import { InspectionTable } from './components/InspectionTable'
import { EscalationQueue } from './components/EscalationQueue'
import { LivePipelineControls } from './components/LivePipelineControls'
import { DefectParetoChart, SeverityPieChart, LatencyTrendChart } from './components/Charts'
import { RunSetup }      from './components/RunSetup'
import { ReportsPage }   from './components/ReportsPage'
import { ProductsPage }  from './components/ProductsPage'
import { SettingsPage }  from './components/SettingsPage'
import { ToastContainer } from './components/Toast'
import { InspectionDetailModal } from './components/InspectionDetailModal'
import type { LatencyTrend } from './types'

function PageHeader({
  title, subtitle, children,
}: { title: string; subtitle?: string; children?: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between flex-wrap gap-3 mb-6">
      <div>
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="page-sub">{subtitle}</p>}
      </div>
      {children && <div className="flex items-center gap-2">{children}</div>}
    </div>
  )
}

function LiveClock() {
  const [time, setTime] = useState(() => new Date().toLocaleTimeString())
  useEffect(() => {
    const t = setInterval(() => setTime(new Date().toLocaleTimeString()), 1000)
    return () => clearInterval(t)
  }, [])
  return <span suppressHydrationWarning>{time}</span>
}

function InnerApp() {
  const { state, dispatch } = useApp()
  const { auth, analytics } = state
  const [tab, setTab] = useState('dashboard')
  const [latency, setLatency] = useState<LatencyTrend[]>([])
  const [latencyLoading, setLatencyLoading] = useState(false)
  const [detailId, setDetailId] = useState<string | null>(null)

  useEffect(() => {
    getHealth()
      .then((h) => {
        dispatch({ type: 'SET_API_STATUS', payload: 'ok' })
        dispatch({ type: 'SET_MODEL_LOADED', payload: h.model_loaded })
      })
      .catch(() => dispatch({ type: 'SET_API_STATUS', payload: 'error' }))
  }, [dispatch])

  useLiveInspections()

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
    if (tab === 'dashboard') {
      loadAnalytics()
      loadLatency()
    }
  }, [tab, loadAnalytics, loadLatency])

  if (!auth) return <LoginPage />

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100 overflow-hidden">
      <Sidebar tab={tab} onTabChange={setTab} />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <header className="flex items-center justify-between h-[60px] px-6 border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm flex-shrink-0">
          <div className="flex items-center gap-2 text-sm">
            <span className="text-gray-500">VisionFood QAI</span>
            <span className="text-gray-700">/</span>
            <span className="text-gray-200 font-medium capitalize">{tab.replace(/-/g, ' ')}</span>
          </div>
          <div className="flex items-center gap-3 text-xs text-gray-500">
            <LiveClock />
            {tab === 'dashboard' && (
              <button
                onClick={() => { loadAnalytics(); loadLatency() }}
                className="btn-ghost text-xs py-1.5 px-2.5 min-h-0 gap-1.5"
                aria-label="Refresh analytics"
              >
                <RefreshCw size={12} className={analytics.loading ? 'animate-spin' : ''} />
                Refresh
              </button>
            )}
          </div>
        </header>

        <main className="flex-1 overflow-y-auto">
          <div className="max-w-screen-2xl mx-auto p-6">

            {tab === 'dashboard' && (
              <div className="space-y-6 animate-fade-in">
                <PageHeader
                  title="Operations Overview"
                  subtitle="Quality intelligence across all active inspection lines"
                >
                  <select
                    value={analytics.hours}
                    onChange={(e) =>
                      dispatch({ type: 'SET_ANALYTICS_HOURS', payload: Number(e.target.value) })
                    }
                    className="select w-auto text-xs py-1.5"
                    aria-label="Time window"
                  >
                    <option value={1}>Last 1 hour</option>
                    <option value={8}>Last 8 hours</option>
                    <option value={24}>Last 24 hours</option>
                    <option value={168}>Last 7 days</option>
                  </select>
                </PageHeader>
                <KPIRow summary={analytics.summary} loading={analytics.loading} />
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                  <DefectParetoChart data={analytics.pareto} loading={analytics.loading} />
                  <SeverityPieChart data={analytics.severity} loading={analytics.loading} />
                  <div className="md:col-span-2 xl:col-span-1">
                    <LiveFeed compact />
                  </div>
                </div>
                <LatencyTrendChart data={latency} loading={latencyLoading} />
              </div>
            )}

            {tab === 'live' && (
              <div className="space-y-5 animate-fade-in">
                <PageHeader title="Live Monitor" subtitle="Real-time inspection stream from edge devices" />
                <RunSetup />
                <LivePipelineControls />
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
                  <div className="lg:col-span-3"><LiveFeed /></div>
                  <div className="lg:col-span-2"><EscalationQueue /></div>
                </div>
              </div>
            )}

            {tab === 'inspect' && (
              <div className="space-y-5 animate-fade-in">
                <PageHeader title="Manual Inspection" subtitle="Upload an image to run on-demand quality analysis" />
                <RunSetup />
                <div className="max-w-2xl mx-auto"><InspectPanel /></div>
              </div>
            )}

            {tab === 'history' && (
              <div className="animate-fade-in">
                <PageHeader title="Inspection History" subtitle="Browse, filter and export all past inspections" />
                <InspectionTable onRowClick={(id) => setDetailId(id)} />
              </div>
            )}

            {tab === 'escalations' && (
              <div className="animate-fade-in">
                <PageHeader title="Escalation Queue" subtitle="Items requiring human review or verdict override" />
                <div className="max-w-3xl mx-auto"><EscalationQueue fullPage /></div>
              </div>
            )}

            {tab === 'products' && (
              <div className="animate-fade-in"><ProductsPage /></div>
            )}

            {tab === 'reports' && (
              <div className="animate-fade-in">
                <ReportsPage analytics={analytics} latency={latency} latencyLoading={latencyLoading} />
              </div>
            )}

            {tab === 'settings' && (
              <div className="animate-fade-in"><SettingsPage /></div>
            )}

          </div>
        </main>
      </div>

      {detailId && (
        <InspectionDetailModal inspectionId={detailId} onClose={() => setDetailId(null)} />
      )}

      <ToastContainer />
    </div>
  )
}

export default function App() {
  return (
    <AppProvider>
      <InnerApp />
    </AppProvider>
  )
}
