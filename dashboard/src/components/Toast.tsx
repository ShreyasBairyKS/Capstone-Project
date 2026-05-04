import { useEffect, memo } from 'react'
import { CheckCircle, XCircle, AlertTriangle, Info, X } from 'lucide-react'
import { useApp } from '../store'
import type { Toast } from '../store'

const CONFIGS: Record<Toast['type'], {
  icon: React.ElementType
  bg: string
  border: string
  iconColor: string
  titleColor: string
}> = {
  success: {
    icon: CheckCircle,
    bg: 'bg-gray-900',
    border: 'border-brand-500/40',
    iconColor: 'text-brand-400',
    titleColor: 'text-brand-300',
  },
  error: {
    icon: XCircle,
    bg: 'bg-gray-900',
    border: 'border-red-500/40',
    iconColor: 'text-red-400',
    titleColor: 'text-red-300',
  },
  warning: {
    icon: AlertTriangle,
    bg: 'bg-gray-900',
    border: 'border-yellow-500/40',
    iconColor: 'text-yellow-400',
    titleColor: 'text-yellow-300',
  },
  info: {
    icon: Info,
    bg: 'bg-gray-900',
    border: 'border-blue-500/40',
    iconColor: 'text-blue-400',
    titleColor: 'text-blue-300',
  },
}

function ToastItem({ toast }: { toast: Toast }) {
  const { dispatch } = useApp()
  const cfg = CONFIGS[toast.type]
  const Icon = cfg.icon

  useEffect(() => {
    const t = setTimeout(
      () => dispatch({ type: 'DISMISS_TOAST', payload: toast.id }),
      4000,
    )
    return () => clearTimeout(t)
  }, [toast.id, dispatch])

  return (
    <div
      role="alert"
      aria-live="polite"
      className={`
        flex items-start gap-3 px-4 py-3 rounded-xl border shadow-xl
        animate-slide-in-right min-w-[280px] max-w-[360px]
        ${cfg.bg} ${cfg.border}
      `}
    >
      <Icon size={16} className={`flex-shrink-0 mt-0.5 ${cfg.iconColor}`} aria-hidden />
      <div className="flex-1 min-w-0">
        <p className={`text-sm font-semibold leading-tight ${cfg.titleColor}`}>
          {toast.title}
        </p>
        {toast.message && (
          <p className="text-xs text-gray-400 mt-0.5 leading-relaxed">{toast.message}</p>
        )}
      </div>
      <button
        onClick={() => dispatch({ type: 'DISMISS_TOAST', payload: toast.id })}
        className="flex-shrink-0 p-0.5 text-gray-500 hover:text-gray-300 transition-colors rounded"
        aria-label="Dismiss"
      >
        <X size={13} />
      </button>
    </div>
  )
}

export const ToastContainer = memo(function ToastContainer() {
  const { state } = useApp()
  const { toasts } = state

  if (toasts.length === 0) return null

  return (
    <div
      aria-label="Notifications"
      className="fixed bottom-5 right-5 z-[100] flex flex-col gap-2 items-end"
    >
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} />
      ))}
    </div>
  )
})
