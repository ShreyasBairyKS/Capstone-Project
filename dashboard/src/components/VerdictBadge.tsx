import type { Verdict } from '../types'

const CONFIG: Record<Verdict, { bg: string; text: string; ring: string; label: string }> = {
  PASS:     { bg: 'bg-green-500/20',  text: 'text-green-400',  ring: 'ring-green-500/50',  label: '✓ PASS' },
  FAIL:     { bg: 'bg-red-500/20',    text: 'text-red-400',    ring: 'ring-red-500/50',    label: '✗ FAIL' },
  ESCALATE: { bg: 'bg-orange-500/20', text: 'text-orange-400', ring: 'ring-orange-500/50', label: '⚠ ESCALATE' },
  REVIEW:   { bg: 'bg-yellow-500/20', text: 'text-yellow-400', ring: 'ring-yellow-500/50', label: '? REVIEW' },
}

interface Props {
  verdict: Verdict
  size?: 'sm' | 'md' | 'lg' | 'xl'
}

export function VerdictBadge({ verdict, size = 'md' }: Props) {
  const cfg = CONFIG[verdict]
  const sizeClass =
    size === 'sm'  ? 'text-xs px-2 py-0.5' :
    size === 'lg'  ? 'text-base px-4 py-1.5 font-bold' :
    size === 'xl'  ? 'text-2xl px-6 py-2 font-extrabold tracking-wider' :
                    'text-sm px-3 py-1 font-semibold'
  return (
    <span
      className={`inline-flex items-center rounded-full ring-1 select-none ${cfg.bg} ${cfg.text} ${cfg.ring} ${sizeClass}`}
      aria-label={`Verdict: ${verdict}`}
      role="status"
    >
      {cfg.label}
    </span>
  )
}

