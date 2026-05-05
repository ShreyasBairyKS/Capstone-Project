import type { SeverityGrade } from '../types'

const CFG: Record<SeverityGrade, { bg: string; text: string; label: string }> = {
  S1: { bg: 'bg-green-500/20',  text: 'text-green-400',  label: 'S1 Minor' },
  S2: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', label: 'S2 Moderate' },
  S3: { bg: 'bg-orange-500/20', text: 'text-orange-400', label: 'S3 Serious' },
  S4: { bg: 'bg-red-500/20',    text: 'text-red-400',    label: 'S4 Critical' },
}

interface Props {
  grade: SeverityGrade
  score?: number
  size?: 'sm' | 'md'
}

export function SeverityBadge({ grade, score, size = 'md' }: Props) {
  const cfg = CFG[grade]
  const sz = size === 'sm' ? 'text-xs px-2 py-0.5' : 'text-sm px-3 py-1 font-semibold'
  return (
    <span className={`inline-flex items-center gap-1 rounded-full ring-1 ring-current/30 ${cfg.bg} ${cfg.text} ${sz}`}>
      {cfg.label}
      {score !== undefined && (
        <span className="opacity-70 text-xs">({score.toFixed(2)})</span>
      )}
    </span>
  )
}
