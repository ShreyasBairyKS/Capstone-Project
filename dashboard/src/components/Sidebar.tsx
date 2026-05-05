import { memo, useState } from 'react'
import {
  LayoutDashboard, Activity, Search, History,
  AlertOctagon, FileText, Settings, Package,
  ChevronLeft, ChevronRight, LogOut,
  Wifi, WifiOff, Cpu, CheckCircle, AlertTriangle,
} from 'lucide-react'
import { useApp } from '../store'

interface NavItem {
  id: string
  label: string
  icon: React.ElementType
  roles?: string[]
  isBadge?: boolean
}

const PRIMARY_NAV: NavItem[] = [
  { id: 'dashboard',   label: 'Dashboard',    icon: LayoutDashboard },
  { id: 'live',        label: 'Live Monitor', icon: Activity },
  { id: 'inspect',     label: 'Inspect',      icon: Search },
  { id: 'history',     label: 'History',      icon: History },
  { id: 'escalations', label: 'Escalations',  icon: AlertOctagon, isBadge: true },
]

const SECONDARY_NAV: NavItem[] = [
  { id: 'products', label: 'Products', icon: Package,   roles: ['supervisor', 'admin'] },
  { id: 'reports',  label: 'Reports',  icon: FileText,  roles: ['supervisor', 'admin'] },
  { id: 'settings', label: 'Settings', icon: Settings,  roles: ['supervisor', 'admin'] },
]

interface Props {
  tab: string
  onTabChange: (tab: string) => void
}

export const Sidebar = memo(function Sidebar({ tab, onTabChange }: Props) {
  const { state, dispatch } = useApp()
  const { auth, apiStatus, modelLoaded, wsConnected, escalationQueue } = state
  const [collapsed, setCollapsed] = useState(false)

  const canSee = (item: NavItem) =>
    !item.roles || (auth && item.roles.includes(auth.role))

  function NavButton({ item }: { item: NavItem }) {
    const active = tab === item.id
    const badge = item.isBadge ? escalationQueue.length : 0
    const Icon = item.icon

    return (
      <button
        onClick={() => onTabChange(item.id)}
        aria-current={active ? 'page' : undefined}
        title={collapsed ? item.label : undefined}
        className={`nav-item relative ${active ? 'nav-active' : 'nav-inactive'} ${
          collapsed ? 'justify-center px-2' : ''
        }`}
      >
        <Icon size={16} className="flex-shrink-0" aria-hidden />
        {!collapsed && <span className="flex-1 text-left truncate">{item.label}</span>}
        {badge > 0 && !collapsed && (
          <span className="bg-orange-500 text-white text-[10px] font-bold rounded-full min-w-[18px] h-[18px] flex items-center justify-center px-1 leading-none">
            {badge > 99 ? '99+' : badge}
          </span>
        )}
        {badge > 0 && collapsed && (
          <span className="absolute top-1 right-1 w-2 h-2 bg-orange-500 rounded-full" />
        )}
      </button>
    )
  }

  const statusItems = [
    {
      label: 'API',
      icon: apiStatus === 'ok'
        ? <CheckCircle size={12} />
        : apiStatus === 'error'
          ? <AlertTriangle size={12} />
          : <span className="w-3 h-3 border-2 border-yellow-400 border-t-transparent rounded-full animate-spin inline-block" />,
      value: apiStatus === 'ok' ? 'Connected' : apiStatus === 'error' ? 'Error' : 'Checking',
      color: apiStatus === 'ok' ? 'text-brand-400' : apiStatus === 'error' ? 'text-red-400' : 'text-yellow-400',
    },
    {
      label: 'Live Stream',
      icon: wsConnected ? <Wifi size={12} /> : <WifiOff size={12} />,
      value: wsConnected ? 'Live' : 'Offline',
      color: wsConnected ? 'text-brand-400' : 'text-gray-500',
    },
    {
      label: 'AI Model',
      icon: <Cpu size={12} />,
      value: modelLoaded ? 'Loaded' : 'Not loaded',
      color: modelLoaded ? 'text-blue-400' : 'text-yellow-400',
    },
  ]

  return (
    <aside
      className={`relative flex flex-col bg-gray-900 border-r border-gray-800 flex-shrink-0
        transition-[width] duration-200 ease-in-out
        ${collapsed ? 'w-[64px]' : 'w-[240px]'}`}
    >
      {/* Brand header */}
      <div
        className={`flex items-center h-[60px] px-4 border-b border-gray-800 flex-shrink-0 gap-3
          ${collapsed ? 'justify-center px-2' : ''}`}
      >
        <div className="w-8 h-8 bg-brand-500 rounded-xl flex items-center justify-center font-extrabold text-gray-950 text-xs flex-shrink-0 shadow-glow-green">
          VF
        </div>
        {!collapsed && (
          <div className="min-w-0 animate-fade-in">
            <p className="font-bold text-white text-sm leading-none tracking-tight">VisionFood</p>
            <p className="text-brand-400 text-[11px] font-medium leading-none mt-1">QAI Platform</p>
          </div>
        )}
      </div>

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed((c) => !c)}
        className="absolute -right-3 top-[72px] w-6 h-6 bg-gray-800 border border-gray-700 rounded-full
          flex items-center justify-center text-gray-400 hover:text-white hover:bg-gray-700
          z-10 transition-colors shadow-md"
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {collapsed ? <ChevronRight size={11} /> : <ChevronLeft size={11} />}
      </button>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-2 space-y-0.5">
        {PRIMARY_NAV.filter(canSee).map((item) => (
          <NavButton key={item.id} item={item} />
        ))}

        {SECONDARY_NAV.filter(canSee).length > 0 && (
          <>
            <div className={`my-2 border-t border-gray-800 ${collapsed ? 'mx-2' : 'mx-1'}`} />
            {SECONDARY_NAV.filter(canSee).map((item) => (
              <NavButton key={item.id} item={item} />
            ))}
          </>
        )}
      </nav>

      {/* System status */}
      {!collapsed && (
        <div className="mx-2 mb-2 p-3 rounded-xl bg-gray-800/50 border border-gray-700/50 space-y-2">
          <p className="text-[10px] text-gray-500 uppercase tracking-widest font-semibold px-1">
            System Status
          </p>
          {statusItems.map(({ label, icon, value, color }) => (
            <div key={label} className="flex items-center gap-2 px-1">
              <span className={color}>{icon}</span>
              <span className="text-gray-400 text-xs flex-1">{label}</span>
              <span className={`text-[11px] font-medium ${color}`}>{value}</span>
            </div>
          ))}
        </div>
      )}

      {/* Collapsed status dots */}
      {collapsed && (
        <div className="mb-3 flex flex-col items-center gap-1.5">
          <span
            className={apiStatus === 'ok' ? 'dot-online' : apiStatus === 'error' ? 'dot-error' : 'dot-offline'}
            title={`API: ${apiStatus}`}
          />
          <span
            className={wsConnected ? 'dot-online' : 'dot-offline'}
            title={wsConnected ? 'Live stream active' : 'Offline'}
          />
          <span
            className={modelLoaded ? 'dot-online' : 'dot-offline'}
            title={modelLoaded ? 'Model loaded' : 'Model not loaded'}
          />
        </div>
      )}

      {/* User section */}
      {auth && (
        <div
          className={`px-3 py-3 border-t border-gray-800 flex items-center gap-2 flex-shrink-0
            ${collapsed ? 'justify-center' : ''}`}
        >
          <div className="w-7 h-7 rounded-full bg-brand-600/30 border border-brand-500/40 flex items-center justify-center text-xs font-bold text-brand-300 flex-shrink-0 uppercase">
            {auth.username[0]}
          </div>
          {!collapsed && (
            <>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-gray-200 font-medium truncate leading-none">{auth.username}</p>
                <p className="text-[11px] text-gray-500 capitalize mt-0.5">{auth.role}</p>
              </div>
              <button
                onClick={() => dispatch({ type: 'SET_AUTH', payload: null })}
                className="btn-icon text-gray-500 hover:text-red-400 hover:bg-red-500/10 border-0"
                aria-label="Sign out"
                title="Sign out"
              >
                <LogOut size={14} />
              </button>
            </>
          )}
        </div>
      )}
    </aside>
  )
})
