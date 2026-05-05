/**
 * RunSetup.tsx
 *
 * Persistent banner shown at the top of the Live and Inspect tabs.
 *
 * No active run  → compact banner with "Start Run" button (opens modal)
 * Active run     → info bar showing product name, SKU, sub-type badge,
 *                  container contents badge, elapsed duration timer,
 *                  inspection / defect counts, and "End Run" button
 *                  (supervisor / admin only; disabled with tooltip for operators)
 *
 * Auto-refreshes every 10 seconds via GET /api/v1/runs/active.
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { getActiveRun, getProducts, startRun, endRun } from '../api'
import { useApp } from '../store'
import type { ProductionRun, Product } from '../types'

const SUBTYPE_COLOUR: Record<string, string> = {
  transparent_bottle: 'bg-cyan-900/60 text-cyan-300 border-cyan-700',
  rigid_can: 'bg-slate-700/60 text-slate-300 border-slate-600',
  flexible_wrapper: 'bg-amber-900/60 text-amber-300 border-amber-700',
  rigid_box: 'bg-orange-900/60 text-orange-300 border-orange-700',
}

const SUBTYPE_LABEL: Record<string, string> = {
  transparent_bottle: 'Transparent Bottle',
  rigid_can: 'Rigid Can',
  flexible_wrapper: 'Flexible Wrapper',
  rigid_box: 'Rigid Box',
}

function formatElapsed(startedAt: string): string {
  const seconds = Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000)
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

// ---------------------------------------------------------------------------
// Start Run Modal
// ---------------------------------------------------------------------------

interface StartRunModalProps {
  onClose: () => void
  onStarted: (run: ProductionRun) => void
  userRole: string
  username: string
}

function StartRunModal({ onClose, onStarted, userRole: _role, username }: StartRunModalProps) {
  const [products, setProducts] = useState<Product[]>([])
  const [loadingProducts, setLoadingProducts] = useState(true)
  const [selectedSku, setSelectedSku] = useState('')
  const [operatorId, setOperatorId] = useState(username)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    getProducts(0, 200)
      .then(setProducts)
      .catch(() => setProducts([]))
      .finally(() => setLoadingProducts(false))
  }, [])

  const handleStart = async () => {
    if (!selectedSku) { setError('Please select a product SKU.'); return }
    setSubmitting(true)
    setError('')
    try {
      const run = await startRun(selectedSku, operatorId || undefined)
      onStarted(run)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to start run.'
      setError(msg.includes('409') ? 'An active run already exists for this SKU.' : msg)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 w-full max-w-sm shadow-2xl">
        <h3 className="text-base font-semibold text-gray-100 mb-1">Start Production Run</h3>
        <p className="text-xs text-gray-500 mb-4">Select the product SKU for this run.</p>

        <div className="space-y-3">
          <div>
            <label htmlFor="run-sku-select" className="block text-xs font-medium text-gray-400 mb-1">
              Product SKU <span className="text-red-400">*</span>
            </label>
            <select
              id="run-sku-select"
              value={selectedSku}
              onChange={(e) => { setSelectedSku(e.target.value); setError('') }}
              disabled={loadingProducts}
              className="w-full bg-gray-800 text-gray-100 text-sm rounded-lg px-3 py-2.5 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            >
              <option value="">{loadingProducts ? 'Loading products…' : 'Select product…'}</option>
              {products.map((p) => (
                <option key={p.sku} value={p.sku}>
                  {p.name} ({p.sku})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="run-operator" className="block text-xs font-medium text-gray-400 mb-1">
              Operator ID
            </label>
            <input
              id="run-operator"
              type="text"
              value={operatorId}
              onChange={(e) => setOperatorId(e.target.value)}
              className="w-full bg-gray-800 text-gray-100 text-sm rounded-lg px-3 py-2.5 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {error && (
            <p className="text-red-400 text-xs bg-red-900/30 border border-red-700 rounded-lg px-3 py-2">
              {error}
            </p>
          )}
        </div>

        <div className="flex gap-3 mt-5">
          <button
            type="button"
            onClick={onClose}
            id="start-run-cancel"
            className="flex-1 py-2.5 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleStart}
            id="start-run-confirm"
            disabled={submitting || loadingProducts}
            className="flex-1 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white text-sm font-semibold rounded-lg transition-colors"
          >
            {submitting ? 'Starting…' : 'Start Run'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main RunSetup component
// ---------------------------------------------------------------------------

interface RunSetupProps {
  /** Product look-up map cached by parent — optional, for displaying product metadata */
  productCache?: Record<string, Product>
}

export function RunSetup({ productCache }: RunSetupProps) {
  const { state, dispatch } = useApp()
  const { auth, activeRun } = state
  const userRole = auth?.role ?? 'operator'
  const username = auth?.username ?? 'operator'
  const isSupervisor = userRole === 'supervisor' || userRole === 'admin'

  const [elapsed, setElapsed] = useState('')
  const [showModal, setShowModal] = useState(false)
  const [ending, setEnding] = useState(false)
  const [error, setError] = useState('')
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Fetch active run every 10 s ──────────────────────────────────────────
  const fetchRun = useCallback(async () => {
    try {
      const run = await getActiveRun()
      dispatch({ type: 'SET_ACTIVE_RUN', payload: run })
    } catch {
      // Silently fail — don't block the inspection UI on DB errors
    }
  }, [dispatch])

  useEffect(() => {
    fetchRun()
    const interval = setInterval(fetchRun, 10_000)
    return () => clearInterval(interval)
  }, [fetchRun])

  // ── Elapsed timer (tick every second when run is active) ─────────────────
  useEffect(() => {
    if (!activeRun) {
      setElapsed('')
      if (timerRef.current) clearInterval(timerRef.current)
      return
    }
    const tick = () => setElapsed(formatElapsed(activeRun.started_at))
    tick()
    timerRef.current = setInterval(tick, 1_000)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [activeRun])

  // ── End run ───────────────────────────────────────────────────────────────
  const handleEndRun = async (status: 'completed' | 'aborted') => {
    if (!activeRun) return
    setEnding(true)
    setError('')
    try {
      await endRun(activeRun.run_id, status)
      dispatch({ type: 'CLEAR_ACTIVE_RUN' })
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to end run.')
    } finally {
      setEnding(false)
    }
  }

  // ── Run started callback ──────────────────────────────────────────────────
  const handleStarted = (run: ProductionRun) => {
    dispatch({ type: 'SET_ACTIVE_RUN', payload: run })
    setShowModal(false)
  }

  // ── Product metadata from cache ───────────────────────────────────────────
  const product = activeRun ? productCache?.[activeRun.sku] : undefined

  // ── No active run — compact banner ────────────────────────────────────────
  if (!activeRun) {
    return (
      <>
        <div
          id="run-setup-no-run"
          className="flex items-center justify-between bg-gray-900/80 border border-gray-800 rounded-xl px-4 py-3 mb-4"
        >
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-gray-600 inline-block" />
            <span className="text-sm text-gray-400">No active production run</span>
          </div>
          <button
            type="button"
            id="start-run-btn"
            onClick={() => setShowModal(true)}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-lg transition-colors"
          >
            Start Run
          </button>
        </div>

        {showModal && (
          <StartRunModal
            onClose={() => setShowModal(false)}
            onStarted={handleStarted}
            userRole={userRole}
            username={username}
          />
        )}
      </>
    )
  }

  // ── Active run — info bar ─────────────────────────────────────────────────
  const subTypeBadge = product?.product_sub_type ?? activeRun.sku
  const badgeClass = SUBTYPE_COLOUR[product?.product_sub_type ?? ''] ?? 'bg-gray-800 text-gray-400 border-gray-700'

  return (
    <div
      id="run-setup-active"
      className="bg-emerald-950/40 border border-emerald-800/50 rounded-xl px-4 py-3 mb-4"
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        {/* Status indicator */}
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block animate-pulse" />
          <span className="text-sm font-semibold text-emerald-300">Active Run</span>
        </div>

        {/* Product info */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-200">{product?.name ?? activeRun.sku}</span>
          <span className="text-xs text-gray-500">({activeRun.sku})</span>
        </div>

        {/* Sub-type badge */}
        {product?.product_sub_type && (
          <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${badgeClass}`}>
            {SUBTYPE_LABEL[subTypeBadge] ?? subTypeBadge}
          </span>
        )}

        {/* Container contents badge */}
        {product?.container_contents && (
          <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${
            product.container_contents === 'liquid'
              ? 'bg-blue-900/40 text-blue-300 border-blue-700'
              : 'bg-stone-800/60 text-stone-300 border-stone-600'
          }`}>
            {product.container_contents}
          </span>
        )}

        {/* Counters */}
        <div className="flex items-center gap-3 ml-auto text-xs">
          <span className="text-gray-400">
            <span className="text-gray-200 font-semibold">{activeRun.inspection_count}</span> inspections
          </span>
          <span className={`font-semibold ${activeRun.defect_count > 0 ? 'text-red-400' : 'text-gray-400'}`}>
            <span className="font-semibold">{activeRun.defect_count}</span> defects
          </span>
          <span className="text-gray-500 tabular-nums">{elapsed}</span>
        </div>

        {/* End Run controls */}
        <div className="flex items-center gap-2">
          {error && <span className="text-xs text-red-400">{error}</span>}
          {isSupervisor ? (
            <>
              <button
                type="button"
                id="end-run-complete-btn"
                onClick={() => handleEndRun('completed')}
                disabled={ending}
                className="px-3 py-1.5 bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition-colors"
              >
                {ending ? '…' : 'End Run'}
              </button>
              <button
                type="button"
                id="end-run-abort-btn"
                onClick={() => handleEndRun('aborted')}
                disabled={ending}
                title="Abort run"
                className="px-3 py-1.5 bg-red-900/60 hover:bg-red-800 disabled:opacity-50 text-red-300 text-xs font-semibold rounded-lg transition-colors border border-red-700"
              >
                Abort
              </button>
            </>
          ) : (
            <span
              title="Supervisor or Admin role required to end a run"
              className="text-xs text-gray-600 cursor-not-allowed px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg"
            >
              End Run ⓘ
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
