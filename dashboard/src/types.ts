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
