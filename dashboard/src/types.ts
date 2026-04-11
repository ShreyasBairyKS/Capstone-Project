// Types matching core/schemas.py

export type Verdict = 'PASS' | 'FAIL' | 'ESCALATE' | 'REVIEW'
export type SeverityGrade = 'S1' | 'S2' | 'S3' | 'S4'
export type RemediationActionType = 'RELABEL' | 'REFILL' | 'REPACK' | 'CLEAN' | 'REJECT' | 'PASS'

export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface Detection {
  class_id: number
  class_name: string
  confidence: number
  bbox: BoundingBox
  bbox_area_ratio: number
}

export interface UQResult {
  mean_confidence: number
  std_confidence: number
  ci_low: number
  ci_high: number
  is_uncertain: boolean
  escalation_required: boolean
  n_passes: number
}

export interface SeverityResult {
  grade: SeverityGrade
  score: number
  area_component: number
  conf_uncertainty_component: number
  class_risk_component: number
  attempt_penalty_component: number
}

export interface RemediationAction {
  action: RemediationActionType
  station: string | null
  is_remediable: boolean
  reason: string
  max_attempts: number
}

export interface LabelQRStatus {
  qr_detected: boolean
  qr_decoded: string | null
  qr_expected: string | null
  qr_matched: boolean | null
  label_anomaly_types: string[]  // e.g. ['misalignment', 'curl', 'wrinkle']
}

export interface ModelVersion {
  model_name: string
  version: string
  active: boolean
  loaded_at: string
  mAP50?: number
}

export interface DeviceStatus {
  device_id: string
  online: boolean
  last_heartbeat: string
  queue_depth: number
}

export interface InspectionResult {
  inspection_id: string
  product_id: string | null
  sku: string
  timestamp: string
  verdict: Verdict
  escalated: boolean
  detections: Detection[]
  uq_result: UQResult | null
  severity_result: SeverityResult | null
  remediation_action: RemediationAction | null
  // Optional: annotated image with bounding boxes drawn (base64)
  annotated_image_b64: string | null
  label_qr: LabelQRStatus | null
  latency_ms: number
  device_id: string
}

export interface InspectionSummary {
  id: string
  product_id: string | null
  sku: string
  timestamp: string
  verdict: Verdict
  escalated: boolean
  latency_ms: number
  device_id: string
  defect_count: number
}

export interface AnalyticsSummary {
  total_inspections: number
  by_verdict: Record<Verdict, number>
  defect_rate: number
  avg_latency_ms: number
  window_hours: number
}

export interface DefectPareto {
  class_name: string
  count: number
  pct: number
}

export interface SeverityDistribution {
  grade: SeverityGrade
  count: number
}

export interface LatencyTrend {
  timestamp: string
  p50_ms: number
  p95_ms: number
  p99_ms: number
}

export interface AuditLogEntry {
  id: string
  timestamp: string
  user: string
  action: string
  target_id: string
  reason: string | null
}

export interface OverrideRequest {
  inspection_id: string
  new_verdict: Verdict
  reason: string
  operator: string
}

