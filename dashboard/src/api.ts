import axios from 'axios'
import type {
  InspectionResult,
  InspectionSummary,
  AnalyticsSummary,
  DefectPareto,
  SeverityDistribution,
} from './types'

// The Vite dev server proxies /api/* → http://localhost:8000/*
// In production, set VITE_API_BASE to the deployed API URL.
const BASE = import.meta.env.VITE_API_BASE ?? '/api'
const API_KEY = import.meta.env.VITE_API_KEY ?? 'dev-insecure-key'

const client = axios.create({
  baseURL: BASE,
  headers: { 'X-API-Key': API_KEY },
})

// --------------------------------------------------------------------------
// Inspections
// --------------------------------------------------------------------------

export async function submitInspection(
  imageB64: string,
  sku = 'default',
  productId?: string,
): Promise<InspectionResult> {
  const { data } = await client.post<InspectionResult>('/inspections', {
    image_b64: imageB64,
    sku,
    product_id: productId ?? null,
    attempt_count: 0,
  })
  return data
}

export async function listInspections(
  limit = 50,
  offset = 0,
  verdict?: string,
): Promise<InspectionSummary[]> {
  const params: Record<string, string | number> = { limit, offset }
  if (verdict) params.verdict = verdict
  const { data } = await client.get<InspectionSummary[]>('/inspections', { params })
  return data
}

export async function getInspection(id: string): Promise<InspectionResult> {
  const { data } = await client.get<InspectionResult>(`/inspections/${id}`)
  return data
}

// --------------------------------------------------------------------------
// Analytics
// --------------------------------------------------------------------------

export async function getAnalyticsSummary(hours = 24): Promise<AnalyticsSummary> {
  const { data } = await client.get<AnalyticsSummary>('/analytics/summary', {
    params: { hours },
  })
  return data
}

export async function getDefectPareto(hours = 24): Promise<DefectPareto[]> {
  const { data } = await client.get<DefectPareto[]>('/analytics/defect-pareto', {
    params: { hours },
  })
  return data
}

export async function getSeverityDistribution(hours = 24): Promise<SeverityDistribution[]> {
  const { data } = await client.get<SeverityDistribution[]>(
    '/analytics/severity-distribution',
    { params: { hours } },
  )
  return data
}

// --------------------------------------------------------------------------
// Health
// --------------------------------------------------------------------------

export async function getHealth(): Promise<{ status: string; model_loaded: boolean }> {
  const { data } = await axios.get(`${BASE}/health`)
  return data
}
