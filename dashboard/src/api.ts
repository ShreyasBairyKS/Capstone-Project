import type {
  InspectionResult,
  InspectionSummary,
  AnalyticsSummary,
  DefectPareto,
  SeverityDistribution,
  LatencyTrend,
  AuditLogEntry,
  OverrideRequest,
  ModelVersion,
  DeviceStatus,
} from './types'

// The Vite dev server proxies /api/* → http://localhost:8000/*
// In production, set VITE_API_BASE to the deployed API URL.
const BASE = import.meta.env.VITE_API_BASE ?? '/api'
const API_KEY = import.meta.env.VITE_API_KEY ?? 'dev-insecure-key'

async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
  params?: Record<string, string | number | boolean>,
): Promise<T> {
  let url = `${BASE}${path}`
  if (params && Object.keys(params).length > 0) {
    const qs = new URLSearchParams(
      Object.entries(params)
        .filter(([, v]) => v !== '' && v !== undefined)
        .map(([k, v]) => [k, String(v)]),
    ).toString()
    if (qs) url = `${url}?${qs}`
  }
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
      ...(options.headers ?? {}),
    },
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`API ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

// --------------------------------------------------------------------------
// Inspections
// --------------------------------------------------------------------------

export async function submitInspection(
  imageB64: string,
  sku = 'default',
  productId?: string,
): Promise<InspectionResult> {
  return apiFetch<InspectionResult>('/inspections', {
    method: 'POST',
    body: JSON.stringify({
      image_b64: imageB64,
      sku,
      product_id: productId ?? null,
      attempt_count: 0,
    }),
  })
}

export interface InspectionListParams {
  limit?: number
  offset?: number
  verdict?: string
  sku?: string
  device_id?: string
  date_from?: string
  date_to?: string
  escalated_only?: boolean
}

export async function listInspections(
  p: InspectionListParams = {},
): Promise<InspectionSummary[]> {
  const { limit = 50, offset = 0, ...rest } = p
  return apiFetch<InspectionSummary[]>('/inspections', {}, { limit, offset, ...rest })
}

export async function getInspection(id: string): Promise<InspectionResult> {
  return apiFetch<InspectionResult>(`/inspections/${id}`)
}

export async function overrideVerdict(req: OverrideRequest): Promise<void> {
  await apiFetch<unknown>(`/inspections/${req.inspection_id}/override`, {
    method: 'POST',
    body: JSON.stringify(req),
  })
}

export async function retryInspection(id: string): Promise<InspectionResult> {
  return apiFetch<InspectionResult>(`/inspections/${id}/retry`, { method: 'POST' })
}

export async function acknowledgeEscalation(id: string): Promise<void> {
  await apiFetch<unknown>(`/inspections/${id}/acknowledge`, { method: 'POST' })
}

// --------------------------------------------------------------------------
// Analytics
// --------------------------------------------------------------------------

export async function getAnalyticsSummary(hours = 24): Promise<AnalyticsSummary> {
  return apiFetch<AnalyticsSummary>('/analytics/summary', {}, { hours })
}

export async function getDefectPareto(hours = 24): Promise<DefectPareto[]> {
  return apiFetch<DefectPareto[]>('/analytics/defect-pareto', {}, { hours })
}

export async function getSeverityDistribution(hours = 24): Promise<SeverityDistribution[]> {
  return apiFetch<SeverityDistribution[]>('/analytics/severity-distribution', {}, { hours })
}

export async function getLatencyTrend(hours = 24): Promise<LatencyTrend[]> {
  return apiFetch<LatencyTrend[]>('/analytics/latency-trend', {}, { hours })
}

// --------------------------------------------------------------------------
// Export
// --------------------------------------------------------------------------

export async function exportInspections(format: 'csv' | 'json', hours = 24): Promise<Blob> {
  const url = `${BASE}/export/inspections?format=${format}&hours=${hours}`
  const res = await fetch(url, { headers: { 'X-API-Key': API_KEY } })
  if (!res.ok) throw new Error(`Export failed: ${res.status}`)
  return res.blob()
}

export async function downloadReport(type: 'daily' | 'weekly'): Promise<Blob> {
  const res = await fetch(`${BASE}/reports/download?type=${type}`, {
    headers: { 'X-API-Key': API_KEY },
  })
  if (!res.ok) throw new Error(`Report download failed: ${res.status}`)
  return res.blob()
}

// --------------------------------------------------------------------------
// Audit log
// --------------------------------------------------------------------------

export async function getAuditLog(limit = 100): Promise<AuditLogEntry[]> {
  return apiFetch<AuditLogEntry[]>('/audit-log', {}, { limit })
}

// --------------------------------------------------------------------------
// Models
// --------------------------------------------------------------------------

export async function getModelVersions(): Promise<ModelVersion[]> {
  return apiFetch<ModelVersion[]>('/models/versions')
}

// --------------------------------------------------------------------------
// Devices
// --------------------------------------------------------------------------

export async function getDeviceStatus(): Promise<DeviceStatus[]> {
  return apiFetch<DeviceStatus[]>('/devices/status')
}

// --------------------------------------------------------------------------
// Health
// --------------------------------------------------------------------------

export async function getHealth(): Promise<{ status: string; model_loaded: boolean }> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`)
  return res.json()
}

