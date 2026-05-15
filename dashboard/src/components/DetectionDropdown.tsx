import { useState } from 'react'
import { DEFECT_CLASS_LABELS, type Detection, type DefectClass } from '../types'
import { ChevronDown, ChevronUp, AlertTriangle } from 'lucide-react'

const CLASS_COLORS: Record<string, string> = {
  improper_filling:     '#3b82f6',
  fill_level_low:       '#3b82f6',
  fill_level_high:      '#ef4444',
  packaging_damage:     '#f97316',
  cap_fitting_anomaly:  '#06b6d4',
  label_misalignment:   '#a855f7',
  surface_contamination:'#ef4444',
  surface_tear:         '#f97316',
  surface_smudge:       '#eab308',
  label_date_mismatch:  '#a855f7',
  label_barcode_mismatch:'#f43f5e',
}

interface Props {
  detections: Detection[]
}

function classLabel(className: string) {
  return DEFECT_CLASS_LABELS[className as DefectClass] ?? className.replace(/_/g, ' ')
}

/**
 * Collapsible dropdown showing the count badge when collapsed,
 * full defect list when expanded.
 */
export function DetectionDropdown({ detections }: Props) {
  const [open, setOpen] = useState(false)

  if (detections.length === 0) {
    return <span className="text-gray-500 text-sm">No defects detected</span>
  }

  return (
    <div className="rounded-lg border border-gray-700 overflow-hidden">
      {/* Toggle button */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-3 py-2 bg-gray-800 hover:bg-gray-700 transition-colors text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          <AlertTriangle size={14} className="text-orange-400" />
          <span className="text-gray-200 font-medium">
              {detections.length} defect{detections.length !== 1 ? 's' : ''} detected
          </span>
          {/* class chips */}
          {!open && detections.slice(0, 3).map((d, i) => (
            <span
              key={i}
              className="text-xs px-1.5 py-0.5 rounded"
              style={{
                background: `${CLASS_COLORS[d.class_name] ?? '#6b7280'}22`,
                color: CLASS_COLORS[d.class_name] ?? '#9ca3af',
              }}
            >
                {classLabel(d.class_name)}
            </span>
          ))}
        </div>
        {open ? <ChevronUp size={14} className="text-gray-400" /> : <ChevronDown size={14} className="text-gray-400" />}
      </button>

      {/* Expanded list */}
      {open && (
        <ul className="divide-y divide-gray-700/50 bg-gray-800/60" role="list">
          {detections.map((det, i) => (
            <li key={i} className="flex items-center justify-between px-3 py-2 text-sm">
              <div className="flex items-center gap-2">
                <span
                  className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ background: CLASS_COLORS[det.class_name] ?? '#6b7280' }}
                  aria-hidden
                />
                <span className="text-gray-200 capitalize">
                  {classLabel(det.class_name)}
                </span>
              </div>
              <div className="flex items-center gap-3 text-xs font-mono">
                <span className="text-red-400">{(det.confidence * 100).toFixed(1)}%</span>
                <span className="text-gray-500">
                  area {(det.bbox_area_ratio * 100).toFixed(1)}%
                </span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
