import { useEffect, useState } from 'react'
import { Cpu, SlidersHorizontal } from 'lucide-react'
import { getLiveInspectionSettings, updateLiveInspectionSettings } from '../api'
import type { LiveInspectionSettings, PipelineMode } from '../types'

const DEFAULT_SETTINGS: LiveInspectionSettings = {
  pipeline_mode: 'standard',
  use_cap_classifier: true,
}

export function LivePipelineControls() {
  const [settings, setSettings] = useState<LiveInspectionSettings>(DEFAULT_SETTINGS)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getLiveInspectionSettings()
      .then(setSettings)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : 'Failed to load settings'))
      .finally(() => setLoading(false))
  }, [])

  async function update(patch: Partial<LiveInspectionSettings>) {
    const next = { ...settings, ...patch }
    setSettings(next)
    setSaving(true)
    setError(null)
    try {
      const saved = await updateLiveInspectionSettings(patch)
      setSettings(saved)
    } catch (err: unknown) {
      setSettings(settings)
      setError(err instanceof Error ? err.message : 'Failed to save settings')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="bg-gray-900 rounded-xl p-4 md:p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <SlidersHorizontal size={15} className="text-blue-400" aria-hidden />
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
            Live Pipeline
          </h3>
        </div>
        <span className={`text-xs ${saving ? 'text-blue-300' : loading ? 'text-gray-500' : 'text-emerald-300'}`}>
          {saving ? 'Saving' : loading ? 'Loading' : 'Ready'}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_auto] gap-3 items-center">
        <div className="grid grid-cols-2 gap-2" role="group" aria-label="Live pipeline mode">
          {([
            ['standard', 'Standard QA'],
            ['yolo_fill_level', 'YOLO + Fill'],
          ] as [PipelineMode, string][]).map(([value, label]) => (
            <button
              key={value}
              type="button"
              disabled={loading || saving}
              onClick={() => update({ pipeline_mode: value })}
              className={`rounded-lg px-3 py-2 text-xs font-semibold transition-colors border min-h-[38px] ${
                settings.pipeline_mode === value
                  ? 'bg-blue-600 text-white border-blue-500'
                  : 'bg-gray-800 text-gray-400 border-gray-700 hover:text-gray-200'
              } disabled:opacity-60`}
            >
              {label}
            </button>
          ))}
        </div>

        <label className={`flex items-center justify-between gap-3 rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm min-h-[38px] ${
          settings.pipeline_mode === 'yolo_fill_level' ? 'text-gray-200' : 'text-gray-600'
        }`}>
          <span className="flex items-center gap-2">
            <Cpu size={13} aria-hidden />
            Cap classifier
          </span>
          <input
            type="checkbox"
            checked={settings.use_cap_classifier}
            disabled={loading || saving || settings.pipeline_mode !== 'yolo_fill_level'}
            onChange={(e) => update({ use_cap_classifier: e.target.checked })}
            className="h-4 w-4 rounded border-gray-600 bg-gray-900 text-blue-600 focus:ring-blue-500 disabled:opacity-40"
          />
        </label>
      </div>

      {error && (
        <p role="alert" className="mt-3 text-red-400 text-xs bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {error}
        </p>
      )}
    </div>
  )
}
