import { useEffect, useState, memo } from 'react'
import { getDeviceStatus } from '../api'
import type { DeviceStatus } from '../types'
import { Monitor, WifiOff } from 'lucide-react'

export const DeviceStatusPanel = memo(function DeviceStatusPanel() {
  const [devices, setDevices] = useState<DeviceStatus[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getDeviceStatus()
      .then(setDevices)
      .catch(() => {/* silent */})
      .finally(() => setLoading(false))
    const t = setInterval(() => {
      getDeviceStatus().then(setDevices).catch(() => {/* silent */})
    }, 30_000)
    return () => clearInterval(t)
  }, [])

  return (
    <div className="bg-gray-900 rounded-xl p-4 md:p-5">
      <div className="flex items-center gap-2 mb-4">
        <Monitor size={14} className="text-purple-400" aria-hidden />
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Device Status</h3>
      </div>

      {loading ? (
        <div className="space-y-2">{[...Array(2)].map((_, i) => (
          <div key={i} className="h-10 bg-gray-800 rounded-lg animate-pulse" />
        ))}</div>
      ) : devices.length === 0 ? (
        <p className="text-gray-600 text-sm text-center py-4">No devices registered</p>
      ) : (
        <ul className="space-y-2">
          {devices.map((d) => (
            <li key={d.device_id} className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm ${d.online ? 'bg-gray-800' : 'bg-red-500/10 border border-red-500/20'}`}>
              {d.online
                ? <span className="w-2 h-2 rounded-full bg-green-400 flex-shrink-0" aria-label="Online" />
                : <WifiOff size={12} className="text-red-400 flex-shrink-0" aria-label="Offline" />}
              <div className="flex-1 min-w-0">
                <p className="text-gray-200 font-mono text-xs truncate">{d.device_id}</p>
                <p className="text-gray-500 text-xs">
                  Last: {new Date(d.last_heartbeat).toLocaleTimeString()}
                </p>
              </div>
              {d.queue_depth > 0 && (
                <span className="text-xs text-orange-400">{d.queue_depth} queued</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
})
