import { useState } from 'react'
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'
import type { Detection } from '../types'

const CLASS_COLORS: Record<string, string> = {
  improper_filling:     '#3b82f6',
  packaging_damage:     '#f97316',
  label_misalignment:   '#a855f7',
  surface_contamination:'#ef4444',
}

interface Props {
  /** If the backend returns an annotated image (base64), show it directly. */
  annotatedB64: string | null
  /** Raw uploaded image preview URL (data URL) — shown with SVG overlay when no annotated image */
  rawImageUrl: string | null
  detections: Detection[]
  /** Natural image dimensions for overlay math (only needed for SVG overlay path) */
  imageWidth?: number
  imageHeight?: number
}

export function BBoxViewer({ annotatedB64, rawImageUrl, detections, imageWidth = 640, imageHeight = 640 }: Props) {
  const [fullscreen, setFullscreen] = useState(false)
  const [zoom, setZoom] = useState(1)

  const src = annotatedB64
    ? `data:image/jpeg;base64,${annotatedB64}`
    : rawImageUrl ?? null

  if (!src) {
    return (
      <div className="w-full h-48 bg-gray-800 rounded-xl flex items-center justify-center text-gray-600 text-sm">
        No image available
      </div>
    )
  }

  const showOverlay = !annotatedB64 && detections.length > 0

  return (
    <>
      {/* Main viewer */}
      <div className="relative bg-gray-800 rounded-xl overflow-hidden border border-gray-700">
        {/* Toolbar */}
        <div className="absolute top-2 right-2 z-10 flex gap-1">
          <button
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
            className="p-1.5 bg-gray-900/80 rounded-lg text-gray-300 hover:text-white transition-colors"
            aria-label="Zoom out"
          >
            <ZoomOut size={14} />
          </button>
          <button
            onClick={() => setZoom((z) => Math.min(3, z + 0.25))}
            className="p-1.5 bg-gray-900/80 rounded-lg text-gray-300 hover:text-white transition-colors"
            aria-label="Zoom in"
          >
            <ZoomIn size={14} />
          </button>
          <button
            onClick={() => setFullscreen(true)}
            className="p-1.5 bg-gray-900/80 rounded-lg text-gray-300 hover:text-white transition-colors"
            aria-label="Fullscreen"
          >
            <Maximize2 size={14} />
          </button>
        </div>

        <div
          className="overflow-auto max-h-96"
          style={{ cursor: zoom > 1 ? 'grab' : 'default' }}
        >
          <div
            className="relative inline-block"
            style={{ transform: `scale(${zoom})`, transformOrigin: 'top left', transition: 'transform 0.15s' }}
          >
            <img
              src={src}
              alt="Inspected product"
              className="max-w-full block"
              draggable={false}
            />

            {/* SVG bounding-box overlay (only when no server-annotated image) */}
            {showOverlay && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox={`0 0 ${imageWidth} ${imageHeight}`}
                preserveAspectRatio="none"
              >
                {detections.map((det, i) => {
                  const { x1, y1, x2, y2 } = det.bbox
                  const color = CLASS_COLORS[det.class_name] ?? '#6b7280'
                  return (
                    <g key={i}>
                      <rect
                        x={x1} y={y1}
                        width={x2 - x1} height={y2 - y1}
                        fill="none"
                        stroke={color}
                        strokeWidth="2"
                      />
                      <rect
                        x={x1} y={Math.max(0, y1 - 18)}
                        width={Math.min(120, x2 - x1)} height="18"
                        fill={color}
                        opacity="0.85"
                      />
                      <text
                        x={x1 + 3}
                        y={Math.max(0, y1 - 3)}
                        fontSize="11"
                        fontFamily="sans-serif"
                        fontWeight="bold"
                        fill="white"
                      >
                        {det.class_name.replace(/_/g, ' ')} {(det.confidence * 100).toFixed(0)}%
                      </text>
                    </g>
                  )
                })}
              </svg>
            )}
          </div>
        </div>

        {zoom !== 1 && (
          <div className="absolute bottom-2 left-2 text-xs bg-gray-900/80 px-2 py-0.5 rounded text-gray-300">
            {Math.round(zoom * 100)}%
          </div>
        )}
      </div>

      {/* Fullscreen overlay */}
      {fullscreen && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
          onClick={() => setFullscreen(false)}
          role="dialog"
          aria-modal
          aria-label="Image fullscreen"
        >
          <button
            className="absolute top-4 right-4 text-white text-3xl leading-none"
            onClick={() => setFullscreen(false)}
            aria-label="Close fullscreen"
          >
            ×
          </button>
          <img
            src={src}
            alt="Inspected product – fullscreen"
            className="max-w-full max-h-full object-contain"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </>
  )
}
