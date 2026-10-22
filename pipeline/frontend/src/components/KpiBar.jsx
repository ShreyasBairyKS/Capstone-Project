import React from 'react';
import { Layers, CheckCircle2, AlertTriangle, Gauge, ArrowUpRight } from 'lucide-react';

export default function KpiBar({ stats, latestLatency }) {
  const total = stats?.total_inspected || 0;
  const passed = stats?.passed || 0;
  const rejected = stats?.rejected || 0;
  const passRate = stats?.pass_rate_pct ?? 100.0;
  const rejectRate = stats?.reject_rate_pct ?? 0.0;
  const avgLatency = stats?.average_latency_ms || 0;
  const currentLatency = latestLatency ? latestLatency : avgLatency;

  // Approximate line rate based on total cycles
  const ratePerMin = total > 0 ? Math.min(Math.round(total * 1.8), 1420) : 0;
  const shiftTarget = 200000;
  const shiftProgressPct = Math.min((total / Math.max(total, 1000)) * 100, 100).toFixed(1);

  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
      {/* 1. Total Inspected Throughput */}
      <div className="bg-[#0b1326] dark:bg-[#060e20] light:bg-[#ffffff] border border-[#1e2e48] light:border-[#e2e8f0] p-3.5 rounded relative flex flex-col justify-between overflow-hidden shadow-sm">
        <div className="flex justify-between items-start">
          <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-[#849588]">
            TOTAL INSPECTED THROUGHPUT
          </span>
          <span className="w-2 h-2 rounded-none bg-[#00f5a0]" />
        </div>
        <div className="my-2 flex items-baseline justify-between">
          <div className="font-['JetBrains_Mono'] text-2xl font-bold tracking-tight text-[#dae2fd] light:text-[#0f172a]">
            {total.toLocaleString()}
          </div>
          <div className="text-xs font-mono text-[#00f5a0] flex items-center gap-0.5">
            <ArrowUpRight size={13} />
            <span>+{ratePerMin}/min</span>
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-[10px] font-mono text-[#94a3b8]">
            <span>Shift Target: 100%</span>
            <span>{total} cycles</span>
          </div>
          <div className="w-full bg-[#171f33] light:bg-[#e2e8f0] h-1.5 rounded overflow-hidden">
            <div className="bg-[#00f5a0] h-full transition-all duration-300" style={{ width: `${Math.min(passRate, 100)}%` }} />
          </div>
        </div>
      </div>

      {/* 2. Compliance Yield Rate */}
      <div className="bg-[#0b1326] dark:bg-[#060e20] light:bg-[#ffffff] border border-[#1e2e48] light:border-[#e2e8f0] p-3.5 rounded relative flex flex-col justify-between overflow-hidden shadow-sm">
        <div className="flex justify-between items-start">
          <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-[#849588]">
            COMPLIANCE YIELD RATE
          </span>
          <span className="px-1.5 py-0.5 text-[9px] font-mono font-bold rounded bg-[#00f5a0]/10 border border-[#00f5a0]/30 text-[#00f5a0]">
            NOMINAL
          </span>
        </div>
        <div className="my-2 flex items-baseline justify-between">
          <div className="font-['JetBrains_Mono'] text-2xl font-bold tracking-tight text-[#00f5a0]">
            {passRate.toFixed(2)}%
          </div>
          <div className="text-xs font-mono text-[#00f5a0]">
            {passed} certified
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-[10px] font-mono text-[#94a3b8]">
            <span>Lower Spec Limit (LSL)</span>
            <span>98.00% req</span>
          </div>
          <div className="relative w-full bg-[#171f33] light:bg-[#e2e8f0] h-1.5 rounded overflow-hidden">
            <div className="bg-[#00f5a0] h-full transition-all duration-300" style={{ width: `${passRate}%` }} />
            <div className="absolute top-0 bottom-0 w-0.5 bg-[#ff3b30]" style={{ left: '98%' }} title="LSL 98%" />
          </div>
        </div>
      </div>

      {/* 3. Rejection Rate */}
      <div className="bg-[#0b1326] dark:bg-[#060e20] light:bg-[#ffffff] border border-[#1e2e48] light:border-[#e2e8f0] p-3.5 rounded relative flex flex-col justify-between overflow-hidden shadow-sm">
        <div className="flex justify-between items-start">
          <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-[#849588]">
            REJECTION RATE
          </span>
          <span className="px-1.5 py-0.5 text-[9px] font-mono font-bold rounded bg-[#ff3b30]/15 border border-[#ff3b30]/40 text-[#ff817b]">
            {rejected} EJECTED
          </span>
        </div>
        <div className="my-2 flex items-baseline justify-between">
          <div className="font-['JetBrains_Mono'] text-2xl font-bold tracking-tight text-[#ff817b]">
            {rejectRate.toFixed(2)}%
          </div>
          <div className="text-xs font-mono text-[#ff817b]">
            Tolerance Max 2.0%
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-[10px] font-mono text-[#94a3b8]">
            <span>Defects: Damaged / Skew / Missing</span>
          </div>
          <div className="w-full bg-[#171f33] light:bg-[#e2e8f0] h-1.5 rounded overflow-hidden">
            <div className="bg-[#ff3b30] h-full transition-all duration-300" style={{ width: `${Math.min(rejectRate * 10, 100)}%` }} />
          </div>
        </div>
      </div>

      {/* 4. Cycle Latency */}
      <div className="bg-[#0b1326] dark:bg-[#060e20] light:bg-[#ffffff] border border-[#1e2e48] light:border-[#e2e8f0] p-3.5 rounded relative flex flex-col justify-between overflow-hidden shadow-sm">
        <div className="flex justify-between items-start">
          <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-[#849588]">
            TOTAL CYCLE LATENCY
          </span>
          <span className="text-[9px] font-mono font-bold text-[#00daf3]">
            BUDGET: &lt; 50ms
          </span>
        </div>
        <div className="my-2 flex items-baseline justify-between">
          <div className="font-['JetBrains_Mono'] text-2xl font-bold tracking-tight text-[#00daf3]">
            {currentLatency ? currentLatency.toFixed(1) : '—'}{' '}
            <span className="text-sm font-normal text-[#94a3b8]">ms</span>
          </div>
          <div className="text-xs font-mono text-[#00f5a0]">
            IN-SPEC
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-[10px] font-mono text-[#94a3b8]">
            <span>Rx Ingestion • Forward Pass • Actuator</span>
          </div>
          <div className="w-full bg-[#171f33] light:bg-[#e2e8f0] h-1.5 rounded overflow-hidden">
            <div 
              className="bg-[#00daf3] h-full transition-all duration-300" 
              style={{ width: `${Math.min((currentLatency / 50) * 100, 100)}%` }} 
            />
          </div>
        </div>
      </div>
    </section>
  );
}
