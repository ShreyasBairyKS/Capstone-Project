import type { InspectionResult } from '../types'
import { VerdictBadge } from './VerdictBadge'
import { AlertTriangle, CheckCircle, Wifi } from 'lucide-react'

interface Props {
  result: InspectionResult | null
  connected: boolean
}

export function LiveFeed({ result, connected }: Props) {
  return (
    <div className="bg-gray-900 rounded-xl p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          Live Inspection Feed
        </h3>
        <div className="flex items-center gap-2 text-xs">
          <Wifi
            size={14}
            className={connected ? 'text-green-400' : 'text-red-400'}
          />
          <span className={connected ? 'text-green-400' : 'text-gray-500'}>
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {!result ? (
        <div className="py-16 text-center text-gray-600">
          <Wifi size={32} className="mx-auto mb-3 opacity-40" />
          <p className="text-sm">Waiting for inspections…</p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Verdict */}
          <div className="flex items-center gap-3">
            <VerdictBadge verdict={result.verdict} size="lg" />
            {result.escalated && (
              <span className="text-xs bg-orange-500/20 text-orange-400 ring-1 ring-orange-500/30 rounded-full px-2 py-0.5">
                Escalated
              </span>
            )}
          </div>

          {/* Meta */}
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <p className="text-gray-500 text-xs">Product</p>
              <p className="text-gray-200 font-mono">{result.product_id ?? '—'}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">SKU</p>
              <p className="text-gray-200">{result.sku}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Latency</p>
              <p className="text-gray-200">{result.latency_ms.toFixed(0)} ms</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Device</p>
              <p className="text-gray-200">{result.device_id}</p>
            </div>
          </div>

          {/* Detections */}
          {result.detections.length > 0 && (
            <div>
              <p className="text-gray-500 text-xs mb-2">Detections ({result.detections.length})</p>
              <div className="space-y-1.5">
                {result.detections.map((det, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between bg-gray-800 rounded-lg px-3 py-2 text-sm"
                  >
                    <span className="text-gray-300">{det.class_name.replace(/_/g, ' ')}</span>
                    <span className="text-red-400 font-mono text-xs">
                      {(det.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* UQ */}
          {result.uq_result && (
            <div className="bg-gray-800 rounded-lg px-3 py-2 text-xs">
              <p className="text-gray-500 mb-1">Uncertainty Quantification</p>
              <div className="flex gap-4 text-gray-300">
                <span>μ = {result.uq_result.mean_confidence.toFixed(3)}</span>
                <span>σ = {result.uq_result.std_confidence.toFixed(3)}</span>
                {result.uq_result.is_uncertain && (
                  <span className="text-yellow-400">⚠ Uncertain</span>
                )}
              </div>
            </div>
          )}

          {/* Severity + Action */}
          {result.severity_result && result.remediation_action && (
            <div className="bg-gray-800 rounded-lg px-3 py-2 text-xs">
              <p className="text-gray-500 mb-1">REMEDY Action</p>
              <div className="flex gap-4 text-gray-300 flex-wrap">
                <span>Grade: <span className="font-semibold text-orange-400">{result.severity_result.grade}</span></span>
                <span>Score: {result.severity_result.score.toFixed(3)}</span>
                <span>Action: <span className="font-semibold text-blue-400">{result.remediation_action.action}</span></span>
                {result.remediation_action.station && (
                  <span>Station {result.remediation_action.station}</span>
                )}
              </div>
              <p className="text-gray-500 mt-1 italic">{result.remediation_action.reason}</p>
            </div>
          )}

          {/* Timestamp */}
          <p className="text-gray-600 text-xs text-right">
            {new Date(result.timestamp).toLocaleTimeString()}
          </p>
        </div>
      )}
    </div>
  )
}
