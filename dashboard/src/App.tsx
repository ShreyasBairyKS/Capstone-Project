import { useEffect, useState, useCallback } from 'react'
import { getAnalyticsSummary, getDefectPareto, getSeverityDistribution, getHealth } from './api'
import type { AnalyticsSummary, DefectPareto, SeverityDistribution } from './types'
import { useLiveStream } from './useLiveStream'
import { KPIRow } from './components/KPIRow'
import { LiveFeed } from './components/LiveFeed'
import { InspectPanel } from './components/InspectPanel'
import { DefectParetoChart, SeverityPieChart } from './components/Charts'
import { InspectionTable } from './components/InspectionTable'
import { Activity, LayoutDashboard, Search, RefreshCw, Wifi, WifiOff } from 'lucide-react'

type Tab = 'dashboard' | 'inspect' | 'history'

const WS_URL = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000/ws/live'

export default function App() {
  const [tab, setTab] = useState<Tab>('dashboard')
  const [hours, setHours] = useState(24)
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null)
  const [pareto, setPareto] = useState<DefectPareto[]>([])
  const [severity, setSeverity] = useState<SeverityDistribution[]>([])
  const [analyticsLoading, setAnalyticsLoading] = useState(true)
  const [apiStatus, setApiStatus] = useState<'ok' | 'error' | 'checking'>('checking')

  const { latest, connected } = useLiveStream(WS_URL)

  const loadAnalytics = useCallback(async () => {
    setAnalyticsLoading(true)
    try {
      const [s, p, sv] = await Promise.all([
        getAnalyticsSummary(hours),
        getDefectPareto(hours),
        getSeverityDistribution(hours),
      ])
      setSummary(s)
      setPareto(p)
      setSeverity(sv)
      setApiStatus('ok')
    } catch {
      setApiStatus('error')
    } finally {
      setAnalyticsLoading(false)
    }
  }, [hours])

  useEffect(() => {
    getHealth()
      .then(() => setApiStatus('ok'))
      .catch(() => setApiStatus('error'))
  }, [])

  useEffect(() => {
    if (tab === 'dashboard') loadAnalytics()
  }, [tab, loadAnalytics])

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top nav */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-3 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-green-500 rounded-lg flex items-center justify-center font-bold text-black text-sm">
            VF
          </div>
          <span className="font-semibold text-gray-100 text-sm">VisionFood QAI</span>
          <span className="text-gray-600 text-xs hidden sm:block">Quality Intelligence Dashboard</span>
        </div>

        <div className="flex items-center gap-4">
          {/* API status */}
          <div className="flex items-center gap-1.5 text-xs">
            {apiStatus === 'ok' ? (
              <><Wifi size={13} className="text-green-400" /><span className="text-green-400">API OK</span></>
            ) : apiStatus === 'error' ? (
              <><WifiOff size={13} className="text-red-400" /><span className="text-red-400">API offline</span></>
            ) : (
              <span className="text-gray-500">Checking…</span>
            )}
          </div>

          {/* Nav tabs */}
          <nav className="flex gap-1">
            {([
              { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
              { id: 'inspect', label: 'Inspect', icon: Search },
              { id: 'history', label: 'History', icon: Activity },
            ] as { id: Tab; label: string; icon: React.ElementType }[]).map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                  tab === id
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`}
              >
                <Icon size={14} />
                <span className="hidden sm:inline">{label}</span>
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full">

        {/* -------------------------------------------------------- */}
        {/* DASHBOARD TAB                                             */}
        {/* -------------------------------------------------------- */}
        {tab === 'dashboard' && (
          <div className="space-y-6">
            {/* Controls */}
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-100">Overview</h2>
              <div className="flex items-center gap-3">
                <select
                  value={hours}
                  onChange={(e) => setHours(Number(e.target.value))}
                  className="bg-gray-800 text-gray-300 rounded-lg px-3 py-1.5 text-xs border border-gray-700 focus:outline-none"
                >
                  <option value={1}>Last 1h</option>
                  <option value={8}>Last 8h</option>
                  <option value={24}>Last 24h</option>
                  <option value={168}>Last 7d</option>
                </select>
                <button
                  onClick={loadAnalytics}
                  className="p-1.5 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-400 transition-colors"
                  title="Refresh"
                >
                  <RefreshCw size={14} className={analyticsLoading ? 'animate-spin' : ''} />
                </button>
              </div>
            </div>

            {/* KPIs */}
            <KPIRow summary={summary} loading={analyticsLoading} />

            {/* Charts + live feed */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <DefectParetoChart data={pareto} loading={analyticsLoading} />
              <SeverityPieChart data={severity} loading={analyticsLoading} />
              <LiveFeed result={latest} connected={connected} />
            </div>
          </div>
        )}

        {/* -------------------------------------------------------- */}
        {/* INSPECT TAB                                              */}
        {/* -------------------------------------------------------- */}
        {tab === 'inspect' && (
          <div className="max-w-md mx-auto">
            <h2 className="text-lg font-semibold text-gray-100 mb-4">Manual Inspection</h2>
            <InspectPanel />
          </div>
        )}

        {/* -------------------------------------------------------- */}
        {/* HISTORY TAB                                              */}
        {/* -------------------------------------------------------- */}
        {tab === 'history' && (
          <div>
            <h2 className="text-lg font-semibold text-gray-100 mb-4">Inspection History</h2>
            <InspectionTable />
          </div>
        )}
      </main>

      <footer className="border-t border-gray-800 text-center py-3 text-gray-600 text-xs">
        VisionFood QAI  ·  Capstone Project 2026
      </footer>
    </div>
  )
}
