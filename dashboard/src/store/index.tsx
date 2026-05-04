import React, {
  createContext,
  useContext,
  useReducer,
  useCallback,
  type ReactNode,
} from 'react'
import type { InspectionResult, InspectionSummary, AnalyticsSummary, DefectPareto, SeverityDistribution, ProductionRun } from '../types'

// ─── Auth ─────────────────────────────────────────────────────────────────────
export type UserRole = 'operator' | 'supervisor' | 'admin'

export interface AuthUser {
  username: string
  role: UserRole
  token: string
}

// ─── Toast ────────────────────────────────────────────────────────────────────
export interface Toast {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message?: string
}

// ─── App state ────────────────────────────────────────────────────────────────
export interface AppState {
  auth: AuthUser | null
  // Live stream
  liveLatest: InspectionResult | null
  wsConnected: boolean
  // Escalation queue (unacknowledged escalated items)
  escalationQueue: InspectionResult[]
  // Analytics
  analytics: {
    summary: AnalyticsSummary | null
    pareto: DefectPareto[]
    severity: SeverityDistribution[]
    hours: number
    loading: boolean
  }
  // Inspection history
  history: {
    rows: InspectionSummary[]
    loading: boolean
    filters: HistoryFilters
  }
  // API health
  apiStatus: 'ok' | 'error' | 'checking'
  modelLoaded: boolean
  // Active production run (V2)
  activeRun: ProductionRun | null
  // UI
  sidebarOpen: boolean
  // Toasts
  toasts: Toast[]
}

export interface HistoryFilters {
  verdict: string
  sku: string
  deviceId: string
  dateFrom: string
  dateTo: string
  escalatedOnly: boolean
}

// ─── Actions ──────────────────────────────────────────────────────────────────
export type Action =
  | { type: 'SET_AUTH'; payload: AuthUser | null }
  | { type: 'SET_LIVE'; payload: InspectionResult }
  | { type: 'SET_WS_CONNECTED'; payload: boolean }
  | { type: 'ACK_ESCALATION'; payload: string } // inspection_id
  | { type: 'ADD_ESCALATION'; payload: InspectionResult }
  | { type: 'SET_ANALYTICS_LOADING'; payload: boolean }
  | { type: 'SET_ANALYTICS'; payload: { summary: AnalyticsSummary; pareto: DefectPareto[]; severity: SeverityDistribution[] } }
  | { type: 'SET_ANALYTICS_HOURS'; payload: number }
  | { type: 'SET_HISTORY_LOADING'; payload: boolean }
  | { type: 'SET_HISTORY'; payload: InspectionSummary[] }
  | { type: 'SET_HISTORY_FILTERS'; payload: Partial<HistoryFilters> }
  | { type: 'SET_API_STATUS'; payload: 'ok' | 'error' | 'checking' }
  | { type: 'SET_MODEL_LOADED'; payload: boolean }
  | { type: 'SET_ACTIVE_RUN'; payload: ProductionRun | null }
  | { type: 'CLEAR_ACTIVE_RUN' }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'PUSH_TOAST'; payload: Omit<Toast, 'id'> }
  | { type: 'DISMISS_TOAST'; payload: string }

// ─── Reducer ──────────────────────────────────────────────────────────────────
const initialFilters: HistoryFilters = {
  verdict: '',
  sku: '',
  deviceId: '',
  dateFrom: '',
  dateTo: '',
  escalatedOnly: false,
}

const initialState: AppState = {
  auth: null,
  liveLatest: null,
  wsConnected: false,
  escalationQueue: [],
  analytics: { summary: null, pareto: [], severity: [], hours: 24, loading: false },
  history: { rows: [], loading: false, filters: initialFilters },
  apiStatus: 'checking',
  modelLoaded: false,
  activeRun: null,
  sidebarOpen: false,
  toasts: [],
}

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_AUTH':
      return { ...state, auth: action.payload }

    case 'SET_LIVE': {
      const result = action.payload
      // Auto-add to escalation queue if escalated and not already present
      const alreadyQueued = state.escalationQueue.some(
        (r) => r.inspection_id === result.inspection_id,
      )
      const escalationQueue =
        result.escalated && !alreadyQueued
          ? [result, ...state.escalationQueue].slice(0, 50)
          : state.escalationQueue
      return { ...state, liveLatest: result, escalationQueue }
    }

    case 'SET_WS_CONNECTED':
      return { ...state, wsConnected: action.payload }

    case 'ACK_ESCALATION':
      return {
        ...state,
        escalationQueue: state.escalationQueue.filter(
          (r) => r.inspection_id !== action.payload,
        ),
      }

    case 'ADD_ESCALATION': {
      const exists = state.escalationQueue.some(
        (r) => r.inspection_id === action.payload.inspection_id,
      )
      if (exists) return state
      return {
        ...state,
        escalationQueue: [action.payload, ...state.escalationQueue].slice(0, 50),
      }
    }

    case 'SET_ANALYTICS_LOADING':
      return { ...state, analytics: { ...state.analytics, loading: action.payload } }

    case 'SET_ANALYTICS':
      return {
        ...state,
        analytics: { ...state.analytics, ...action.payload, loading: false },
      }

    case 'SET_ANALYTICS_HOURS':
      return { ...state, analytics: { ...state.analytics, hours: action.payload } }

    case 'SET_HISTORY_LOADING':
      return { ...state, history: { ...state.history, loading: action.payload } }

    case 'SET_HISTORY':
      return { ...state, history: { ...state.history, rows: action.payload, loading: false } }

    case 'SET_HISTORY_FILTERS':
      return {
        ...state,
        history: {
          ...state.history,
          filters: { ...state.history.filters, ...action.payload },
        },
      }

    case 'SET_API_STATUS':
      return { ...state, apiStatus: action.payload }

    case 'SET_MODEL_LOADED':
      return { ...state, modelLoaded: action.payload }

    case 'SET_ACTIVE_RUN':
      return { ...state, activeRun: action.payload }

    case 'CLEAR_ACTIVE_RUN':
      return { ...state, activeRun: null }

    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarOpen: !state.sidebarOpen }

    case 'PUSH_TOAST': {
      const toast: Toast = {
        ...action.payload,
        id: `toast-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      }
      return { ...state, toasts: [...state.toasts, toast].slice(-6) }
    }

    case 'DISMISS_TOAST':
      return { ...state, toasts: state.toasts.filter((t) => t.id !== action.payload) }

    default:
      return state
  }
}

// ─── Context ──────────────────────────────────────────────────────────────────
interface CtxValue {
  state: AppState
  dispatch: React.Dispatch<Action>
}

const Ctx = createContext<CtxValue | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState)
  return <Ctx.Provider value={{ state, dispatch }}>{children}</Ctx.Provider>
}

export function useAppState() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useAppState must be inside AppProvider')
  return ctx.state
}

export function useAppDispatch() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useAppDispatch must be inside AppProvider')
  return ctx.dispatch
}

/** Convenience: both state and dispatch */
export function useApp() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useApp must be inside AppProvider')
  return ctx
}

/** Hook to dispatch toasts easily */
export function useToast() {
  const { dispatch } = useApp()
  return useCallback(
    (type: Toast['type'], title: string, message?: string) =>
      dispatch({ type: 'PUSH_TOAST', payload: { type, title, message } }),
    [dispatch],
  )
}
