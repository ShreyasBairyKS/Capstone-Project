import { memo } from 'react'
import { useAppState } from '../store'
import { VerdictBadge } from './VerdictBadge'
import { SeverityBadge } from './SeverityBadge'
import { DetectionDropdown } from './DetectionDropdown'
import { BBoxViewer } from './BBoxViewer'
import { InferenceSummaryPanel } from './InferenceSummaryPanel'
import { Wifi, WifiOff } from 'lucide-react'

export const LiveFeed = memo(function LiveFeed({ compact }: { compact?: boolean }) {
  const { liveLatest: result, wsConnected: connected } = useAppState()

  return (
    <div className="bg-gray-900 rounded-xl p-4 md:p-5 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          Live Feed
        </h3>
        <div className="flex items-center gap-1.5 text-xs">
          {connected
            ? <><Wifi size={13} className="text-green-400" aria-hidden /><span className="text-green-400">Live</span></>
            : <><WifiOff size={13} className="text-gray-500" aria-hidden /><span className="text-gray-500">Disconnected</span></>}
        </div>
      </div>

      {!result ? (
        <div className="py-12 text-center text-gray-600">
          <Wifi size={28} className="mx-auto mb-3 opacity-30" aria-hidden />
          <p className="text-sm">Waiting for inspectionsâ€¦</p>
        </div>
      ) : (
        <>
          {/* Verdict â€“ large for shop-floor readability */}
          <div className="flex flex-wrap items-center gap-2">
            <VerdictBadge verdict={result.verdict} size="lg" />
            {result.severity_result && (
              <SeverityBadge grade={result.severity_result.grade} score={result.severity_result.score} />
            )}
            {result.escalated && (
              <span className="text-xs bg-orange-500/20 text-orange-400 ring-1 ring-orange-500/30 rounded-full px-2 py-0.5">
                Escalated
              </span>
            )}
          </div>

          {/* Annotated / raw image */}
          <BBoxViewer
            annotatedB64={result.annotated_image_b64 ?? null}
            rawImageUrl={null}
            detections={result.detections}
          />

          {/* Detections dropdown */}
          <DetectionDropdown detections={result.detections} />

          <InferenceSummaryPanel summary={result.inference_summary ?? null} />

          {/* Product meta */}
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
            {([
              ['Product', result.product_id ?? 'â€”'],
              ['SKU', result.sku],
              ['Device', result.device_id],
              ['Latency', `${result.latency_ms.toFixed(0)} ms`],
              ['Time', new Date(result.timestamp).toLocaleTimeString()],
            ] as [string, string][]).map(([label, value]) => (
              <div key={label}>
                <dt className="text-gray-500 text-xs">{label}</dt>
                <dd className="text-gray-200 font-mono text-xs truncate">{value}</dd>
              </div>
            ))}
          </dl>

          {/* UQ */}
          {result.uq_result && (
            <div className="bg-gray-800 rounded-lg px-3 py-2 text-xs space-y-1">
              <p className="text-gray-500 font-medium">Uncertainty (MC Dropout)</p>
              <div className="flex flex-wrap gap-3 text-gray-300">
                <span>Î¼ = {result.uq_result.mean_confidence.toFixed(3)}</span>
                <span>Ïƒ = {result.uq_result.std_confidence.toFixed(3)}</span>
                <span>CI [{result.uq_result.ci_low.toFixed(2)}, {result.uq_result.ci_high.toFixed(2)}]</span>
                {result.uq_result.is_uncertain && (
                  <span className="text-yellow-400 font-semibold">âš  Uncertain</span>
                )}
              </div>
            </div>
          )}

          {/* REMEDY */}
          {result.remediation_action && (
            <div className="bg-gray-800 rounded-lg px-3 py-2 text-xs space-y-1">
              <p className="text-gray-500 font-medium">REMEDY Action</p>
              <div className="flex flex-wrap gap-3 text-gray-300">
                <span>Action: <span className="text-blue-400 font-semibold">{result.remediation_action.action}</span></span>
                {result.remediation_action.station && (
                  <span>â†’ Station {result.remediation_action.station}</span>
                )}
                <span className={result.remediation_action.is_remediable ? 'text-green-400' : 'text-red-400'}>
                  {result.remediation_action.is_remediable ? 'Remediable' : 'Not remediable'}
                </span>
              </div>
              <p className="text-gray-500 italic">{result.remediation_action.reason}</p>
            </div>
          )}

          {/* Label / QR status */}
          {result.label_qr && (
            <div className="bg-gray-800 rounded-lg px-3 py-2 text-xs space-y-1">
              <p className="text-gray-500 font-medium">Label & QR</p>
              <div className="flex flex-wrap gap-3 text-gray-300">
                <span>QR: {result.label_qr.qr_detected
                  ? (result.label_qr.qr_matched === false
                    ? <span className="text-red-400">Mismatch</span>
                    : <span className="text-green-400">Matched</span>)
                  : 'Not detected'}</span>
                {result.label_qr.qr_expected && (
                  <span>Expected: <code className="text-yellow-400">{result.label_qr.qr_expected}</code></span>
                )}
                {result.label_qr.qr_decoded && (
                  <span>Decoded: <code className="text-gray-200">{result.label_qr.qr_decoded}</code></span>
                )}
              </div>
              {result.label_qr.label_anomaly_types.length > 0 && (
                <p className="text-orange-400">
                  Label: {result.label_qr.label_anomaly_types.join(', ')}
                </p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
})
