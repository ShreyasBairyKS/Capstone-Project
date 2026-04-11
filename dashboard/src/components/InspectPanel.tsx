import { useState, useRef, memo } from 'react'
import { Upload, Camera, X } from 'lucide-react'
import { submitInspection } from '../api'
import type { InspectionResult } from '../types'
import { VerdictBadge } from './VerdictBadge'
import { SeverityBadge } from './SeverityBadge'
import { DetectionDropdown } from './DetectionDropdown'
import { BBoxViewer } from './BBoxViewer'

export const InspectPanel = memo(function InspectPanel() {
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
      setError(e instanceof Error ? e.message : 'Request failed')
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
    <div className="bg-gray-900 rounded-xl p-4 md:p-5 space-y-4">
      <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
        Manual Inspection
      </h3>

      {/* Upload zone */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload inspection image"
        className="border-2 border-dashed border-gray-700 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-blue-500 transition-colors min-h-[140px] relative overflow-hidden"
        onClick={() => fileRef.current?.click()}
        onKeyDown={(e) => e.key === 'Enter' && fileRef.current?.click()}
      >
        {imagePreview ? (
          <>
            <img src={imagePreview} alt="preview" className="max-h-40 object-contain" />
            <button
              className="absolute top-2 right-2 bg-gray-800/80 rounded-full p-1 hover:bg-gray-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
              onClick={(e) => { e.stopPropagation(); clear() }}
              aria-label="Remove image"
            >
              <X size={14} />
            </button>
          </>
        ) : (
          <div className="text-center py-8 text-gray-500 pointer-events-none">
            <Upload size={28} className="mx-auto mb-2" aria-hidden />
            <p className="text-sm">Click or drag to upload image</p>
            <p className="text-xs mt-1">JPEG or PNG</p>
          </div>
        )}
        <input ref={fileRef} type="file" accept="image/jpeg,image/png" className="hidden" onChange={onFileChange} />
      </div>

      {/* Options */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-gray-500 mb-1 block" htmlFor="inspect-sku">SKU</label>
          <input id="inspect-sku" type="text" value={sku} onChange={(e) => setSku(e.target.value)}
            className="w-full bg-gray-800 text-gray-200 rounded-lg px-3 py-2 text-sm border border-gray-700 focus:outline-none focus:border-blue-500"
            placeholder="default" />
        </div>
        <div>
          <label className="text-xs text-gray-500 mb-1 block" htmlFor="inspect-pid">Product ID</label>
          <input id="inspect-pid" type="text" value={productId} onChange={(e) => setProductId(e.target.value)}
            className="w-full bg-gray-800 text-gray-200 rounded-lg px-3 py-2 text-sm border border-gray-700 focus:outline-none focus:border-blue-500"
            placeholder="e.g. P00123" />
        </div>
      </div>

      <button
        onClick={runInspection}
        disabled={!imageB64 || loading}
        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 text-white font-semibold rounded-lg py-2.5 text-sm transition-colors flex items-center justify-center gap-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 min-h-[44px]"
      >
        {loading
          ? <span className="animate-spin border-2 border-white border-t-transparent rounded-full w-4 h-4" aria-hidden />
          : <Camera size={16} aria-hidden />}
        {loading ? 'Inspectingâ€¦' : 'Run Inspection'}
      </button>

      {error && (
        <p role="alert" className="text-red-400 text-xs bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {error}
        </p>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-3 border-t border-gray-800 pt-4">
          <div className="flex flex-wrap items-center gap-2">
            <VerdictBadge verdict={result.verdict} size="lg" />
            {result.severity_result && (
              <SeverityBadge grade={result.severity_result.grade} score={result.severity_result.score} />
            )}
            <span className="text-gray-400 text-sm ml-auto">{result.latency_ms.toFixed(0)} ms</span>
          </div>

          {/* Annotated image */}
          <BBoxViewer
            annotatedB64={result.annotated_image_b64 ?? null}
            rawImageUrl={imagePreview}
            detections={result.detections}
          />

          <DetectionDropdown detections={result.detections} />

          {result.remediation_action && (
            <div className="text-xs bg-gray-800 rounded-lg px-3 py-2 text-gray-400 space-y-0.5">
              <p>
                <span className="text-orange-400 font-semibold">{result.severity_result?.grade}</span>
                {' Â· '}
                <span className="text-blue-400">{result.remediation_action.action}</span>
                {result.remediation_action.station ? ` â†’ Station ${result.remediation_action.station}` : ''}
              </p>
              <p className="italic text-gray-500">{result.remediation_action.reason}</p>
            </div>
          )}

          {result.uq_result && (
            <div className="text-xs bg-gray-800 rounded-lg px-3 py-2 text-gray-400">
              Î¼={result.uq_result.mean_confidence.toFixed(3)} Ïƒ={result.uq_result.std_confidence.toFixed(3)}
              {result.uq_result.is_uncertain && (
                <span className="ml-2 text-yellow-400">âš  Uncertain</span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
})
