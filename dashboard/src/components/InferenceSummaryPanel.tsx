import { Activity, Gauge } from 'lucide-react'
import type { InferenceSummary, YoloFillAnnotation } from '../types'

interface Props {
  summary: InferenceSummary | null
}

function pct(value: number | null | undefined) {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'n/a'
}

function ratio(value: number | null | undefined) {
  return typeof value === 'number' ? value.toFixed(3) : 'n/a'
}

function label(value: string | null | undefined) {
  return value ? value.replace(/_/g, ' ') : 'n/a'
}

function AnnotationRow({ annotation }: { annotation: YoloFillAnnotation }) {
  return (
    <li className="px-3 py-2 text-xs">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="text-gray-300 font-medium">
          Bottle {(annotation.bottle_index ?? 0) + 1}
        </span>
        <span className="text-gray-500">
          bottle conf <span className="font-mono text-gray-300">{pct(annotation.bottle_confidence)}</span>
        </span>
        <span className="text-gray-500">
          cap <span className="font-mono text-cyan-300">{label(annotation.cap.verdict)}</span>
        </span>
        <span className="text-gray-500">
          cap det <span className="font-mono text-gray-300">{pct(annotation.cap.detection_confidence)}</span>
        </span>
        <span className="text-gray-500">
          cap cls <span className="font-mono text-gray-300">{pct(annotation.cap.quality_confidence)}</span>
        </span>
        <span className="text-gray-500">
          fill <span className="font-mono text-emerald-300">{label(annotation.fill.level)}</span>
        </span>
        <span className="text-gray-500">
          ratio <span className="font-mono text-gray-300">{ratio(annotation.fill.ratio)}</span>
        </span>
        <span className="text-gray-500">
          water conf <span className="font-mono text-gray-300">{pct(annotation.fill.water_confidence)}</span>
        </span>
      </div>
    </li>
  )
}

export function InferenceSummaryPanel({ summary }: Props) {
  if (!summary) return null

  const annotations = summary.annotations ?? []

  return (
    <div className="rounded-lg border border-gray-700 overflow-hidden bg-gray-800/50">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 border-b border-gray-700/70">
        <div className="flex items-center gap-2 text-sm">
          <Activity size={14} className="text-blue-400" aria-hidden />
          <span className="font-semibold text-gray-200">Inference Details</span>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-gray-400">
          <span>{summary.pipeline.replace(/_/g, ' ')}</span>
          {typeof summary.cap_classifier_enabled === 'boolean' && (
            <span>
              cap classifier{' '}
              <span className={summary.cap_classifier_enabled ? 'text-emerald-300' : 'text-gray-500'}>
                {summary.cap_classifier_enabled ? 'on' : 'off'}
              </span>
            </span>
          )}
          {typeof summary.bottle_count === 'number' && (
            <span>{summary.bottle_count} bottle{summary.bottle_count === 1 ? '' : 's'}</span>
          )}
          <span>P1 {summary.caps_pass1 ?? 0}</span>
          <span>P2 {summary.caps_pass2 ?? 0}</span>
        </div>
      </div>

      {annotations.length > 0 ? (
        <ul className="divide-y divide-gray-700/60">
          {annotations.map((annotation, index) => (
            <AnnotationRow key={annotation.bottle_index ?? index} annotation={annotation} />
          ))}
        </ul>
      ) : (
        <div className="flex items-center gap-2 px-3 py-2 text-xs text-gray-500">
          <Gauge size={13} aria-hidden />
          No per-bottle annotations returned
        </div>
      )}
    </div>
  )
}
