// Types matching core/schemas.py

export type Verdict = 'PASS' | 'FAIL' | 'ESCALATE' | 'REVIEW'
export type SeverityGrade = 'S1' | 'S2' | 'S3' | 'S4'
export type RemediationActionType = 'RELABEL' | 'REFILL' | 'REPACK' | 'CLEAN' | 'REJECT' | 'PASS'

// ─── Product classification types ─────────────────────────────────────────────
export type ProductCategory = 'beverage' | 'food' | 'general'

export type ProductSubType =
  | 'transparent_bottle'
  | 'rigid_can'
  | 'flexible_wrapper'
  | 'rigid_box'

export type ContainerContents = 'liquid' | 'solid'

// ─── Defect classes (original + V2 additions) ─────────────────────────────────
export type DefectClass =
  | 'surface_contamination'
  | 'improper_filling'
  | 'packaging_damage'
  | 'label_misalignment'
  // V2 additions
  | 'fill_level_low'
  | 'fill_level_high'
  | 'cap_fitting_anomaly'
  | 'surface_tear'
  | 'surface_smudge'
  | 'label_date_mismatch'
  | 'label_barcode_mismatch'

export const DEFECT_CLASS_LABELS: Record<DefectClass, string> = {
  surface_contamination: 'Surface Contamination',
  improper_filling: 'Improper Filling',
  packaging_damage: 'Packaging Damage',
  label_misalignment: 'Label Misalignment',
  fill_level_low: 'Underfill',
  fill_level_high: 'Overfill',
  cap_fitting_anomaly: 'Cap Missing/Misfit',
  surface_tear: 'Surface Tear',
  surface_smudge: 'Surface Smudge',
  label_date_mismatch: 'Label Date Mismatch',
  label_barcode_mismatch: 'Label Barcode Mismatch',
}

export const PRODUCT_SUBTYPE_LABELS: Record<ProductSubType, string> = {
  transparent_bottle: 'Transparent Bottle',
  rigid_can: 'Rigid Can',
  flexible_wrapper: 'Flexible Wrapper (Pouch)',
  rigid_box: 'Rigid Box (Carton)',
}

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

export interface LabelTextStatus {
  fields_checked: number
  fields_matched: number
  mismatched_fields: string[]  // field names that failed OCR match
  raw_ocr_results: Record<string, string>  // field_name → extracted text
}

export type PipelineMode = 'standard' | 'yolo_fill_level'

export interface LiveInspectionSettings {
  pipeline_mode: PipelineMode
  use_cap_classifier: boolean
}

export interface YoloFillAnnotation {
  bottle_index: number
  bottle_bbox: number[] | null
  bottle_confidence: number | null
  cap: {
    verdict: string | null
    bbox: number[] | null
    quality: string | null
    quality_confidence: number | null
    detection_confidence: number | null
    detection_source: string | null
  }
  fill: {
    level: string | null
    ratio: number | null
    water_bbox: number[] | null
    water_confidence: number | null
  }
}

export interface InferenceSummary {
  pipeline: string
  cap_classifier_enabled?: boolean
  bottle_count?: number
  caps_pass1?: number
  caps_pass2?: number
  annotations?: YoloFillAnnotation[]
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
  inference_summary: InferenceSummary | null
  label_qr: LabelQRStatus | null
  // V2 additions
  label_text: LabelTextStatus | null
  product_category: ProductCategory | null
  product_sub_type: ProductSubType | null
  container_contents: ContainerContents | null
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

// ─── Product & ProductionRun (MongoDB documents) ──────────────────────────────

export interface ExpectedDateField {
  name: string     // e.g. 'expiry_date', 'mfg_date'
  format: string   // e.g. 'MM/YYYY', 'DD/MM/YYYY'
  value?: string   // expected printed value (optional)
}

export interface Product {
  _id?: string
  sku: string
  name: string
  description?: string | null
  product_category: ProductCategory
  product_sub_type: ProductSubType
  container_contents: ContainerContents
  sku_profile_name: string
  qr_code?: string | null
  expected_dates: ExpectedDateField[]
  created_at: string
  updated_at: string
  __v: number
}

export interface ProductCreate {
  sku: string
  name: string
  description?: string
  product_category: ProductCategory
  product_sub_type: ProductSubType
  container_contents: ContainerContents
  sku_profile_name: string
  qr_code?: string
  expected_dates?: ExpectedDateField[]
}

export interface ProductionRun {
  _id?: string
  run_id: string
  sku: string
  product_id?: string | null
  started_at: string
  ended_at?: string | null
  status: 'active' | 'completed' | 'aborted'
  operator_id?: string | null
  inspection_count: number
  defect_count: number
}
