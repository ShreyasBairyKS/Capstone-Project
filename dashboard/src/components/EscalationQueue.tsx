import { useState, memo } from 'react'
import { useApp } from '../store'
import { acknowledgeEscalation, overrideVerdict } from '../api'
import { VerdictBadge } from './VerdictBadge'
import { DetectionDropdown } from './DetectionDropdown'
import type { Verdict, InspectionResult } from '../types'
import { CheckCheck, RotateCcw, ChevronDown, ChevronUp, AlertOctagon } from 'lucide-react'

// ─── Override dialog ──────────────────────────────────────────────────────────
function OverrideDialog({
  item,
  onClose,
}: {
  item: InspectionResult
  onClose: () => void
}) {
  const [newVerdict, setNewVerdict] = useState<Verdict>('PASS')
  const [reason, setReason] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit() {
    if (!reason.trim()) { setError('Reason is required'); return }
    setLoading(true)
    try {
      await overrideVerdict({ inspection_id: item.inspection_id, new_verdict: newVerdict, reason, operator: 'current-user' })
      onClose()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Override failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4"
      role="dialog"
      aria-modal
      aria-label="Override verdict"
    >
      <div className="bg-gray-900 rounded-xl w-full max-w-md p-6 space-y-4 border border-gray-700">
        <h2 className="text-base font-semibold text-white">Override Verdict</h2>
        <p className="text-gray-400 text-xs font-mono truncate">ID: {item.inspection_id}</p>

        <div className="space-y-1">
          <label className="text-xs text-gray-400 block" htmlFor="new-verdict">New Verdict</label>
          <select
            id="new-verdict"
            value={newVerdict}
            onChange={(e) => setNewVerdict(e.target.value as Verdict)}
            className="w-full bg-gray-800 text-gray-200 rounded-lg px-3 py-2 text-sm border border-gray-700 focus:outline-none focus:border-blue-500"
          >
            {(['PASS', 'FAIL', 'ESCALATE', 'REVIEW'] as Verdict[]).map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <div className="space-y-1">
          <label className="text-xs text-gray-400 block" htmlFor="override-reason">Reason (required)</label>
          <textarea
            id="override-reason"
            rows={3}
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            className="w-full bg-gray-800 text-gray-200 rounded-lg px-3 py-2 text-sm border border-gray-700 focus:outline-none focus:border-blue-500 resize-none"
            placeholder="Explain why you are overriding this verdict…"
          />
        </div>

        {error && <p role="alert" className="text-red-400 text-xs">{error}</p>}

        <div className="flex gap-3 pt-2">
          <button
            onClick={onClose}
            className="flex-1 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg py-2 text-sm transition-colors min-h-[44px]"
          >
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={loading}
            className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white font-semibold rounded-lg py-2 text-sm transition-colors min-h-[44px]"
          >
            {loading ? 'Saving…' : 'Confirm Override'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ─── Queue item card ──────────────────────────────────────────────────────────
function EscalationCard({ item }: { item: InspectionResult }) {
  const { dispatch } = useApp()
  const [expanded, setExpanded] = useState(false)
  const [showOverride, setShowOverride] = useState(false)

  async function ack() {
    try {
      await acknowledgeEscalation(item.inspection_id)
    } catch {
      // optimistic: always dismiss from queue
    }
    dispatch({ type: 'ACK_ESCALATION', payload: item.inspection_id })
  }

  return (
    <>
      <div className="border border-orange-500/30 bg-orange-500/5 rounded-xl p-3 space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <VerdictBadge verdict={item.verdict} size="sm" />
          <span className="text-gray-400 text-xs font-mono truncate max-w-[120px]">{item.product_id ?? item.inspection_id.slice(0, 8)}</span>
          <span className="text-gray-500 text-xs">{item.sku}</span>
          <span className="text-gray-600 text-xs ml-auto">{new Date(item.timestamp).toLocaleTimeString()}</span>
          <button
            onClick={() => setExpanded((o) => !o)}
            className="p-1 text-gray-400 hover:text-gray-200 transition-colors"
            aria-label={expanded ? 'Collapse' : 'Expand'}
          >
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>

        {expanded && (
          <div className="pt-1">
            <DetectionDropdown detections={item.detections} />
            {item.uq_result && (
              <p className="text-xs text-gray-500 mt-1">
                μ={item.uq_result.mean_confidence.toFixed(3)} σ={item.uq_result.std_confidence.toFixed(3)}
                {item.uq_result.is_uncertain && <span className="text-yellow-400 ml-2">⚠ Uncertain</span>}
              </p>
            )}
          </div>
        )}

        <div className="flex gap-2 pt-1">
          <button
            onClick={ack}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-green-600/20 hover:bg-green-600/40 text-green-400 rounded-lg text-xs transition-colors min-h-[36px] flex-1"
          >
            <CheckCheck size={12} aria-hidden /> Acknowledge
          </button>
          <button
            onClick={() => setShowOverride(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600/20 hover:bg-blue-600/40 text-blue-400 rounded-lg text-xs transition-colors min-h-[36px] flex-1"
          >
            <RotateCcw size={12} aria-hidden /> Override
          </button>
        </div>
      </div>

      {showOverride && (
        <OverrideDialog item={item} onClose={() => setShowOverride(false)} />
      )}
    </>
  )
}

// ─── Queue panel ──────────────────────────────────────────────────────────────
export const EscalationQueue = memo(function EscalationQueue() {
  const { state } = useApp()
  const queue = state.escalationQueue

  return (
    <div className="bg-gray-900 rounded-xl p-4 md:p-5">
      <div className="flex items-center gap-2 mb-4">
        <AlertOctagon size={15} className="text-orange-400" aria-hidden />
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          Escalation Queue
        </h3>
        {queue.length > 0 && (
          <span className="ml-auto bg-orange-500 text-black text-xs font-bold rounded-full w-5 h-5 flex items-center justify-center">
            {queue.length}
          </span>
        )}
      </div>

      {queue.length === 0 ? (
        <p className="text-gray-600 text-sm text-center py-8">No pending escalations</p>
      ) : (
        <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
          {queue.map((item) => (
            <EscalationCard key={item.inspection_id} item={item} />
          ))}
        </div>
      )}
    </div>
  )
})
