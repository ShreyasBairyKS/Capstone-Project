import { useState, useRef } from 'react'
import { Upload, Camera, X } from 'lucide-react'
import { submitInspection } from '../api'
import type { InspectionResult } from '../types'
import { VerdictBadge } from './VerdictBadge'

export function InspectPanel() {
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [imageB64, setImageB64] = useState<string | null>(null)
  const [sku, setSku] = useState('default')
  const [productId, setProductId] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<InspectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const dataUrl = ev.target?.result as string
      setImagePreview(dataUrl)
      // Strip the data:...;base64, prefix — API expects raw base64
      setImageB64(dataUrl.split(',')[1])
      setResult(null)
      setError(null)
    }
    reader.readAsDataURL(file)
  }

  async function runInspection() {
    if (!imageB64) return
    setLoading(true)
    setError(null)
    try {
      const res = await submitInspection(imageB64, sku, productId || undefined)
      setResult(res)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Request failed'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  function clear() {
    setImagePreview(null)
    setImageB64(null)
    setResult(null)
    setError(null)
    if (fileRef.current) fileRef.current.value = ''
  }

  return (
    <div className="bg-gray-900 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
        Manual Inspection
      </h3>

      {/* Upload zone */}
      <div
        className="border-2 border-dashed border-gray-700 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-blue-500 transition-colors min-h-[160px] relative overflow-hidden mb-4"
        onClick={() => fileRef.current?.click()}
      >
        {imagePreview ? (
          <>
            <img src={imagePreview} alt="preview" className="max-h-48 object-contain" />
            <button
              className="absolute top-2 right-2 bg-gray-800/80 rounded-full p-1 hover:bg-gray-700"
              onClick={(e) => { e.stopPropagation(); clear() }}
            >
              <X size={14} />
            </button>
          </>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <Upload size={28} className="mx-auto mb-2" />
            <p className="text-sm">Click to upload image</p>
            <p className="text-xs mt-1">JPEG or PNG</p>
          </div>
        )}
        <input
          ref={fileRef}
          type="file"
          accept="image/jpeg,image/png"
          className="hidden"
          onChange={onFileChange}
        />
      </div>

      {/* Options */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className="text-xs text-gray-500 mb-1 block">SKU</label>
          <input
            type="text"
            value={sku}
            onChange={(e) => setSku(e.target.value)}
            className="w-full bg-gray-800 text-gray-200 rounded-lg px-3 py-2 text-sm border border-gray-700 focus:outline-none focus:border-blue-500"
            placeholder="default"
          />
        </div>
        <div>
          <label className="text-xs text-gray-500 mb-1 block">Product ID (optional)</label>
          <input
            type="text"
            value={productId}
            onChange={(e) => setProductId(e.target.value)}
            className="w-full bg-gray-800 text-gray-200 rounded-lg px-3 py-2 text-sm border border-gray-700 focus:outline-none focus:border-blue-500"
            placeholder="e.g. P00123"
          />
        </div>
      </div>

      <button
        onClick={runInspection}
        disabled={!imageB64 || loading}
        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 text-white font-semibold rounded-lg py-2.5 text-sm transition-colors flex items-center justify-center gap-2"
      >
        {loading ? (
          <span className="animate-spin border-2 border-white border-t-transparent rounded-full w-4 h-4" />
        ) : (
          <Camera size={16} />
        )}
        {loading ? 'Inspecting…' : 'Run Inspection'}
      </button>

      {/* Error */}
      {error && (
        <p className="mt-3 text-red-400 text-xs bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {error}
        </p>
      )}

      {/* Result */}
      {result && (
        <div className="mt-4 space-y-3 border-t border-gray-800 pt-4">
          <div className="flex items-center gap-3">
            <VerdictBadge verdict={result.verdict} size="lg" />
            <span className="text-gray-400 text-sm">{result.latency_ms.toFixed(0)} ms</span>
          </div>
          {result.detections.length > 0 && (
            <div className="space-y-1">
              {result.detections.map((d, i) => (
                <div key={i} className="flex justify-between text-sm bg-gray-800 rounded-lg px-3 py-1.5">
                  <span className="text-gray-300">{d.class_name.replace(/_/g, ' ')}</span>
                  <span className="text-red-400 font-mono">{(d.confidence * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
          {result.severity_result && result.remediation_action && (
            <div className="text-xs bg-gray-800 rounded-lg px-3 py-2 text-gray-400">
              <span className="text-orange-400 font-semibold">{result.severity_result.grade}</span>
              {' · '}
              {result.remediation_action.action}
              {result.remediation_action.station ? ` → Station ${result.remediation_action.station}` : ''}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
