import type { Verdict } from '../types'

const VERDICT_CONFIG: Record<Verdict, { bg: string; text: string; ring: string }> = {
  PASS:    { bg: 'bg-green-500/20',  text: 'text-green-400',  ring: 'ring-green-500/50' },
  FAIL:    { bg: 'bg-red-500/20',    text: 'text-red-400',    ring: 'ring-red-500/50' },
  ESCALATE:{ bg: 'bg-orange-500/20', text: 'text-orange-400', ring: 'ring-orange-500/50' },
  REVIEW:  { bg: 'bg-yellow-500/20', text: 'text-yellow-400', ring: 'ring-yellow-500/50' },
}

interface Props {
  verdict: Verdict
  size?: 'sm' | 'md' | 'lg'
}

export function VerdictBadge({ verdict, size = 'md' }: Props) {
  const cfg = VERDICT_CONFIG[verdict]
  const sizeClass = size === 'sm' ? 'text-xs px-2 py-0.5' : size === 'lg' ? 'text-lg px-4 py-1.5 font-bold' : 'text-sm px-3 py-1'
  return (
    <span className={`inline-flex items-center rounded-full ring-1 font-semibold ${cfg.bg} ${cfg.text} ${cfg.ring} ${sizeClass}`}>
      {verdict}
    </span>
  )
}
