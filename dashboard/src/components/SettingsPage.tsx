import { useState, useEffect, useCallback } from 'react'
import { Settings, Cpu, Monitor, Shield, Database, ChevronDown, ChevronUp } from 'lucide-react'
import { ModelInfoPanel } from './ModelInfoPanel'
import { DeviceStatusPanel } from './DeviceStatusPanel'
import { getAuditLog } from '../api'
import type { AuditLogEntry } from '../types'

type SettingsTab = 'system' | 'models' | 'devices' | 'audit'

const SETTINGS_TABS: { id: SettingsTab; label: string; icon: React.ElementType }[] = [
  { id: 'system',  label: 'System',    icon: Settings },
  { id: 'models',  label: 'AI Models', icon: Cpu },
  { id: 'devices', label: 'Devices',   icon: Monitor },
  { id: 'audit',   label: 'Audit Log', icon: Shield },
]

function SystemInfoPanel() {
  const items = [
    { label: 'API URL', value: import.meta.env.VITE_API_URL ?? 'http://localhost:8000' },
    { label: 'Environment', value: import.meta.env.MODE ?? 'development' },
    { label: 'Platform', value: 'VisionFood QAI v1.0' },
    { label: 'Auth Method', value: 'API Key (X-API-Key)' },
    { label: 'WS URL', value: import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000/ws/live' },
  ]
  return (
    <div className="card space-y-1">
      <div className="card-header mb-4">
        <Database size={14} className="text-brand-400" />
        <span>System Configuration</span>
      </div>
      {items.map(({ label, value }) => (
        <div key={label} className="flex items-center gap-3 py-2.5 border-b border-gray-800/60 last:border-0">
          <span className="text-gray-500 text-xs w-36 flex-shrink-0">{label}</span>
          <span className="text-gray-200 text-xs font-mono break-all">{value}</span>
        </div>
      ))}
    </div>
  )
}

function AuditLogPanel() {
  const [entries, setEntries] = useState<AuditLogEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getAuditLog(100)
      setEntries(data)
    } catch {
      setEntries([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  if (loading) {
    return (
      <div className="card space-y-2 animate-pulse">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="h-10 bg-gray-800 rounded-lg" />
        ))}
      </div>
    )
  }

  if (entries.length === 0) {
    return (
      <div className="card flex flex-col items-center justify-center py-12 text-gray-600 gap-2">
        <Shield size={32} />
        <p className="text-sm">No audit entries found</p>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="card-header mb-4">
        <Shield size={14} className="text-brand-400" />
        <span>Audit Log</span>
        <span className="ml-auto text-xs text-gray-500">{entries.length} entries</span>
      </div>
      <div className="space-y-1.5 max-h-[520px] overflow-y-auto pr-1">
        {entries.map((entry, i) => {
          const id = entry.id ?? String(i)
          const isOpen = expanded === id
          return (
            <div key={id} className="border border-gray-800 rounded-xl overflow-hidden">
              <button
                className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-gray-800/40 transition-colors"
                onClick={() => setExpanded(isOpen ? null : id)}
              >
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider ${
                  entry.action?.includes('override') ? 'text-orange-400 bg-orange-500/10' :
                  entry.action?.includes('ack') ? 'text-green-400 bg-green-500/10' :
                  'text-blue-400 bg-blue-500/10'
                }`}>
                  {entry.action ?? 'action'}
                </span>
                <span className="text-gray-400 text-xs flex-1 truncate">{entry.user}</span>
                <span className="text-gray-600 text-xs whitespace-nowrap">
                  {new Date(entry.timestamp).toLocaleString()}
                </span>
                {isOpen ? <ChevronUp size={12} className="text-gray-500 flex-shrink-0" /> : <ChevronDown size={12} className="text-gray-500 flex-shrink-0" />}
              </button>
              {isOpen && (
                <div className="px-4 py-3 bg-gray-900 border-t border-gray-800 text-xs text-gray-400 space-y-1">
                  <p><span className="text-gray-500 mr-2">Target:</span><span className="font-mono">{entry.target_id}</span></p>
                  {entry.reason && (
                    <p><span className="text-gray-500 mr-2">Reason:</span>{entry.reason}</p>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function SettingsPage() {
  const [tab, setTab] = useState<SettingsTab>('system')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="page-title">Settings</h1>
        <p className="page-sub">System configuration, model status, and audit logs</p>
      </div>

      {/* Inner tab bar */}
      <div className="flex gap-1 bg-gray-900 p-1 rounded-xl border border-gray-800 w-fit">
        {SETTINGS_TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              tab === id
                ? 'bg-gray-800 text-white shadow-sm'
                : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/50'
            }`}
          >
            <Icon size={13} />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="animate-fade-in">
        {tab === 'system' && <SystemInfoPanel />}
        {tab === 'models' && (
          <div className="max-w-2xl">
            <ModelInfoPanel />
          </div>
        )}
        {tab === 'devices' && (
          <div className="max-w-2xl">
            <DeviceStatusPanel />
          </div>
        )}
        {tab === 'audit' && <AuditLogPanel />}
      </div>
    </div>
  )
}
