import { useEffect, useState, memo } from 'react'
import { getModelVersions } from '../api'
import type { ModelVersion } from '../types'
import { Cpu, Tag } from 'lucide-react'

export const ModelInfoPanel = memo(function ModelInfoPanel() {
  const [versions, setVersions] = useState<ModelVersion[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getModelVersions()
      .then(setVersions)
      .catch(() => {/* silent */})
      .finally(() => setLoading(false))
  }, [])

  const active = versions.find((v) => v.active)

  return (
    <div className="bg-gray-900 rounded-xl p-4 md:p-5">
      <div className="flex items-center gap-2 mb-4">
        <Cpu size={14} className="text-blue-400" aria-hidden />
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Model Versions</h3>
      </div>

      {loading ? (
        <div className="space-y-2">{[...Array(2)].map((_, i) => (
          <div key={i} className="h-10 bg-gray-800 rounded-lg animate-pulse" />
        ))}</div>
      ) : versions.length === 0 ? (
        <p className="text-gray-600 text-sm text-center py-4">No model info available</p>
      ) : (
        <ul className="space-y-2">
          {versions.map((v) => (
            <li key={v.version} className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm ${v.active ? 'bg-blue-600/15 border border-blue-500/30' : 'bg-gray-800'}`}>
              <Tag size={12} className={v.active ? 'text-blue-400' : 'text-gray-500'} aria-hidden />
              <div className="flex-1 min-w-0">
                <p className="text-gray-200 font-mono truncate">{v.model_name} v{v.version}</p>
                <p className="text-gray-500 text-xs">{new Date(v.loaded_at).toLocaleDateString()}</p>
              </div>
              {v.active && (
                <span className="text-xs bg-blue-500/20 text-blue-400 rounded-full px-2 py-0.5">Active</span>
              )}
              {v.mAP50 !== undefined && (
                <span className="text-xs text-gray-500">mAP {(v.mAP50 * 100).toFixed(0)}%</span>
              )}
            </li>
          ))}
        </ul>
      )}
      {active && (
        <p className="text-xs text-gray-600 mt-3 text-center">Active: {active.model_name} v{active.version}</p>
      )}
    </div>
  )
})
