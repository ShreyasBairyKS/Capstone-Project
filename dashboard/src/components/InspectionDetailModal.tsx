import { useEffect, useCallback, useState } from 'react'
import { X, CheckCircle, XCircle, AlertTriangle, AlertOctagon, Clock, Cpu, Tag, Layers } from 'lucide-react'
import { getInspection } from '../api'
import { VerdictBadge } from './VerdictBadge'
import { BBoxViewer } from './BBoxViewer'
import type { InspectionResult, SeverityGrade } from '../types'

interface Props {
  inspectionId: string
  onClose: () => void
}

const SEVERITY_COLORS: Record<SeverityGrade, string> = {
  S1: 'text-green-400 bg-green-500/10 border-green-500/30',
  S2: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
  S3: 'text-orange-400 bg-orange-500/10 border-orange-500/30',
  S4: 'text-red-400 bg-red-500/10 border-red-500/30',
}

function SeverityBadge({ grade }: { grade: SeverityGrade }) {
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold border ${SEVERITY_COLORS[grade]}`}>
      {grade}
    </span>
  )
}

function DetailRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 py-2 border-b border-gray-800/60 last:border-0">
      <span className="text-gray-500 text-xs w-32 flex-shrink-0 pt-0.5">{label}</span>
      <span className="text-gray-200 text-xs font-medium break-all">{value}</span>
    </div>
  )
}

export function InspectionDetailModal({ inspectionId, onClose }: Props) {
  const [data, setData] = useState<InspectionResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await getInspection(inspectionId)
      setData(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load inspection')
    } finally {
      setLoading(false)
    }
  }, [inspectionId])

  useEffect(() => { load() }, [load])

  // Close on backdrop click
  function handleBackdrop(e: React.MouseEvent<HTMLDivElement>) {
    if (e.target === e.currentTarget) onClose()
  }

  // Close on Escape
  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4"
      role="dialog"
      aria-modal
      aria-label="Inspection details"
      onClick={handleBackdrop}
    >
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-3xl max-h-[90vh] flex flex-col shadow-2xl animate-bounce-in">
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-gray-800 flex-shrink-0">
          <div className="flex-1 min-w-0">
            <h2 className="text-sm font-semibold text-white">Inspection Detail</h2>
            <p className="text-gray-500 text-xs font-mono truncate mt-0.5">{inspectionId}</p>
          </div>
          <button
            onClick={onClose}
            className="btn-icon text-gray-400 hover:text-white border-gray-700 hover:border-gray-600"
            aria-label="Close"
          >
            <X size={14} />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-5">
          {loading && (
            <div className="space-y-3 animate-pulse">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="h-8 bg-gray-800 rounded-lg" />
              ))}
            </div>
          )}

          {error && (
            <div className="flex flex-col items-center justify-center py-12 gap-3 text-red-400">
              <AlertTriangle size={32} />
              <p className="text-sm">{error}</p>
              <button onClick={load} className="btn btn-secondary text-xs">Retry</button>
            </div>
          )}

          {data && !loading && (
            <div className="space-y-6">
              {/* Top badges */}
              <div className="flex flex-wrap items-center gap-3">
                <VerdictBadge verdict={data.verdict} />
                {data.severity_result && (
                  <SeverityBadge grade={data.severity_result.grade} />
                )}
                {data.escalated && (
                  <span className="inline-flex items-center gap-1 text-xs text-orange-400 bg-orange-500/10 border border-orange-500/30 px-2 py-0.5 rounded-full">
                    <AlertOctagon size={11} /> Escalated
                  </span>
                )}
                {data.uq_result?.is_uncertain && (
                  <span className="inline-flex items-center gap-1 text-xs text-yellow-400 bg-yellow-500/10 border border-yellow-500/30 px-2 py-0.5 rounded-full">
                    <AlertTriangle size={11} /> Uncertain
                  </span>
                )}
              </div>

              {/* Image view */}
              {data.annotated_image_b64 && (
                <div className="rounded-xl overflow-hidden border border-gray-800">
                  <BBoxViewer
                    annotatedB64={data.annotated_image_b64 ?? null}
                    rawImageUrl={null}
                    detections={data.detections}
                  />
                </div>
              )}

              {/* Core info */}
              <div className="card-sm">
                <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold mb-3">Inspection Info</p>
                <DetailRow label="Product ID" value={data.product_id ?? '-'} />
                <DetailRow label="SKU" value={data.sku} />
                <DetailRow label="Device" value={data.device_id} />
                <DetailRow label="Timestamp" value={new Date(data.timestamp).toLocaleString()} />
                <DetailRow
                  label="Latency"
                  value={
                    <span className="flex items-center gap-1 text-blue-300">
                      <Clock size={11} />{data.latency_ms.toFixed(1)} ms
                    </span>
                  }
                />
              </div>

              {/* Detections */}
              {data.detections.length > 0 && (
                <div className="card-sm">
                  <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold mb-3 flex items-center gap-2">
                    <Layers size={12} /> Detections ({data.detections.length})
                  </p>
                  <div className="space-y-2">
                    {data.detections.map((d, i) => (
                      <div key={i} className="flex items-center gap-3 text-xs py-2 border-b border-gray-800/50 last:border-0">
                        <span className="text-gray-300 font-medium w-36 truncate">{d.class_name}</span>
                        <div className="flex-1 bg-gray-800 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="h-full bg-brand-500 rounded-full transition-all"
                            style={{ width: `${(d.confidence * 100).toFixed(0)}%` }}
                          />
                        </div>
                        <span className="text-gray-400 w-12 text-right">{(d.confidence * 100).toFixed(1)}%</span>

                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* UQ Result */}
              {data.uq_result && (
                <div className="card-sm">
                  <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold mb-3 flex items-center gap-2">
                    <Cpu size={12} /> Uncertainty Quantification
                  </p>
                  <DetailRow
                    label="Mean Confidence"
                    value={<span className="font-mono">{data.uq_result.mean_confidence.toFixed(4)}</span>}
                  />
                  <DetailRow
                    label="Std Deviation"
                    value={<span className="font-mono">{data.uq_result.std_confidence.toFixed(4)}</span>}
                  />
                  <DetailRow
                    label="Uncertain"
                    value={data.uq_result.is_uncertain
                      ? <span className="text-yellow-400 flex items-center gap-1"><AlertTriangle size={11} /> Yes</span>
                      : <span className="text-green-400 flex items-center gap-1"><CheckCircle size={11} /> No</span>
                    }
                  />
                </div>
              )}

              {/* Severity */}
              {data.severity_result && (
                <div className="card-sm">
                  <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold mb-3">Severity Assessment</p>
                  <DetailRow label="Grade" value={<SeverityBadge grade={data.severity_result.grade} />} />
                  <DetailRow label="Score" value={<span className="font-mono">{data.severity_result.score.toFixed(3)}</span>} />
                  <DetailRow label="Area" value={<span className="font-mono">{data.severity_result.area_component.toFixed(3)}</span>} />
                </div>
              )}

              {/* Remediation */}
              {data.remediation_action && (
                <div className="card-sm border-l-2 border-orange-500/50">
                  <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold mb-3">Remediation</p>
                  <DetailRow label="Action" value={data.remediation_action.action} />
                  {data.remediation_action.reason && (
                    <DetailRow label="Reason" value={data.remediation_action.reason} />
                  )}
                  <DetailRow label="Station" value={data.remediation_action.station ?? '-'} />
                  <DetailRow label="Max Attempts" value={String(data.remediation_action.max_attempts)} />
                </div>
              )}

              {/* QR / Label */}
              {data.label_qr && (
                <div className="card-sm">
                  <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold mb-3 flex items-center gap-2">
                    <Tag size={12} /> Label / QR Data
                  </p>
                  <DetailRow label="QR Detected" value={
                    data.label_qr.qr_detected
                      ? <span className="text-green-400 flex items-center gap-1"><CheckCircle size={11} /> Yes</span>
                      : <span className="text-red-400 flex items-center gap-1"><XCircle size={11} /> No</span>
                  } />
                  {data.label_qr.qr_matched != null && (
                    <DetailRow label="QR Match" value={
                      data.label_qr.qr_matched
                        ? <span className="text-green-400 flex items-center gap-1"><CheckCircle size={11} /> Matched</span>
                        : <span className="text-red-400 flex items-center gap-1"><XCircle size={11} /> Mismatch</span>
                    } />
                  )}
                  {data.label_qr.qr_decoded && (
                    <DetailRow label="Decoded" value={<span className="font-mono text-gray-400 text-xs">{data.label_qr.qr_decoded}</span>} />
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
