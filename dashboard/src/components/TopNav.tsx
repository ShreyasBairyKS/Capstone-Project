import { memo } from 'react'
import { useApp } from '../store'
import { Wifi, WifiOff, AlertOctagon, LogOut } from 'lucide-react'

interface Props {
  tab: string
  onTabChange: (t: string) => void
  tabs: { id: string; label: string; icon: React.ElementType; roles?: string[] }[]
}

export const TopNav = memo(function TopNav({ tab, onTabChange, tabs }: Props) {
  const { state, dispatch } = useApp()
  const { auth, apiStatus, modelLoaded, wsConnected, escalationQueue } = state

  const allowedTabs = tabs.filter(
    (t) => !t.roles || (auth && t.roles.includes(auth.role)),
  )

  return (
    <header className="bg-gray-900 border-b border-gray-800 px-4 md:px-6 py-2 flex items-center gap-3 sticky top-0 z-20 flex-wrap">
      {/* Brand */}
      <div className="flex items-center gap-2 flex-shrink-0 mr-2">
        <div className="w-7 h-7 bg-green-500 rounded-lg flex items-center justify-center font-bold text-black text-xs">
          VF
        </div>
        <span className="font-semibold text-gray-100 text-sm hidden sm:block">VisionFood QAI</span>
      </div>

      {/* Nav tabs */}
      <nav className="flex gap-1 flex-wrap" aria-label="Main navigation">
        {allowedTabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onTabChange(id)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs transition-colors min-h-[36px] focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
              tab === id
                ? 'bg-gray-700 text-white'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
            }`}
            aria-current={tab === id ? 'page' : undefined}
          >
            <Icon size={13} aria-hidden />
            <span className="hidden md:inline">{label}</span>
            {id === 'escalation' && escalationQueue.length > 0 && (
              <span className="bg-orange-500 text-black text-xs font-bold rounded-full w-4 h-4 flex items-center justify-center ml-0.5">
                {escalationQueue.length > 9 ? '9+' : escalationQueue.length}
              </span>
            )}
          </button>
        ))}
      </nav>

      {/* Status indicators */}
      <div className="ml-auto flex items-center gap-3">
        {/* WS */}
        <div className="flex items-center gap-1 text-xs" title="WebSocket stream">
          {wsConnected
            ? <Wifi size={12} className="text-green-400" aria-hidden />
            : <WifiOff size={12} className="text-gray-500" aria-hidden />}
          <span className={wsConnected ? 'text-green-400 hidden sm:block' : 'text-gray-500 hidden sm:block'}>
            {wsConnected ? 'Live' : 'Offline'}
          </span>
        </div>

        {/* API */}
        <div className="flex items-center gap-1 text-xs" title="Backend API">
          <span className={`w-2 h-2 rounded-full ${
            apiStatus === 'ok' ? 'bg-green-400' :
            apiStatus === 'error' ? 'bg-red-400' : 'bg-yellow-400 animate-pulse'
          }`} aria-hidden />
          <span className={`hidden sm:block ${
            apiStatus === 'ok' ? 'text-green-400' :
            apiStatus === 'error' ? 'text-red-400' : 'text-yellow-400'
          }`}>{apiStatus === 'ok' ? 'API OK' : apiStatus === 'error' ? 'API Error' : '…'}</span>
        </div>

        {/* Model */}
        {!modelLoaded && (
          <div className="flex items-center gap-1 text-xs text-yellow-400" title="Model not loaded">
            <AlertOctagon size={12} aria-hidden />
            <span className="hidden sm:block">No model</span>
          </div>
        )}

        {/* User + logout */}
        {auth && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400 hidden md:block capitalize">{auth.username} ({auth.role})</span>
            <button
              onClick={() => dispatch({ type: 'SET_AUTH', payload: null })}
              className="p-1.5 text-gray-500 hover:text-red-400 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500 rounded"
              aria-label="Sign out"
              title="Sign out"
            >
              <LogOut size={13} aria-hidden />
            </button>
          </div>
        )}
      </div>
    </header>
  )
})
