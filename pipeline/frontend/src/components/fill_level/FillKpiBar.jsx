import React from 'react';
import { Activity, CheckCircle2, AlertTriangle, AlertOctagon, Droplets, Clock } from 'lucide-react';
import { useStation } from '../../context/StationContext';

export default function FillKpiBar({ stats, latestLatency }) {
  const { fillHealth } = useStation();
  const underPct = ((fillHealth?.underfill_threshold ?? 0.40) * 100).toFixed(1);
  const overPct = ((fillHealth?.overfill_threshold ?? 0.85) * 100).toFixed(1);
  const total = stats?.total_inspected ?? 0;
  const passCount = stats?.pass_count ?? 0;
  const passRate = stats?.pass_rate ?? 0.0;
  const underfillCount = stats?.underfill_count ?? 0;
  const overfillCount = stats?.overfill_count ?? 0;
  const dryCount = stats?.dry_count ?? 0;
  const meanFillRatio = stats?.mean_fill_ratio ?? 0.0;
  const latency = latestLatency ?? stats?.mean_latency_ms ?? 0;

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-2.5">
      {/* 1. Total Inspected */}
      <div className="p-3 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-col justify-between shadow-sm">
        <div className="flex items-center justify-between text-[#849588] text-[10px] font-mono font-bold tracking-wider">
          <span>TOTAL INSPECTIONS</span>
          <Activity size={14} className="text-[#00daf3]" />
        </div>
        <div className="text-2xl font-mono font-black text-[#dae2fd] mt-1 tracking-tight">
          {total.toLocaleString()}
        </div>
        <div className="text-[10px] font-mono text-[#849588] mt-1 flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-[#00daf3]" />
          <span>STATION 2 CONVEYOR</span>
        </div>
      </div>

      {/* 2. Nominal Pass Yield */}
      <div className="p-3 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-col justify-between shadow-sm">
        <div className="flex items-center justify-between text-[#849588] text-[10px] font-mono font-bold tracking-wider">
          <span>NOMINAL YIELD</span>
          <CheckCircle2 size={14} className="text-[#00f5a0]" />
        </div>
        <div className="text-2xl font-mono font-black text-[#00f5a0] mt-1 tracking-tight">
          {passRate.toFixed(1)}%
        </div>
        <div className="text-[10px] font-mono text-[#849588] mt-1 flex items-center gap-1">
          <span>{passCount.toLocaleString()} PASS / TARGET &ge; 98.0%</span>
        </div>
      </div>

      {/* 3. Underfill Rejects */}
      <div className="p-3 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-col justify-between shadow-sm">
        <div className="flex items-center justify-between text-[#849588] text-[10px] font-mono font-bold tracking-wider">
          <span>UNDERFILL DEFECTS</span>
          <AlertTriangle size={14} className="text-[#f59e0b]" />
        </div>
        <div className="text-2xl font-mono font-black text-[#f59e0b] mt-1 tracking-tight">
          {underfillCount.toLocaleString()}
        </div>
        <div className="text-[10px] font-mono text-[#849588] mt-1">
          &lt; {underPct}% FILL THRESHOLD
        </div>
      </div>

      {/* 4. Overfill Rejects */}
      <div className="p-3 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-col justify-between shadow-sm">
        <div className="flex items-center justify-between text-[#849588] text-[10px] font-mono font-bold tracking-wider">
          <span>OVERFILL DEFECTS</span>
          <AlertOctagon size={14} className="text-[#ff3b30]" />
        </div>
        <div className="text-2xl font-mono font-black text-[#ff3b30] mt-1 tracking-tight">
          {overfillCount.toLocaleString()}
        </div>
        <div className="text-[10px] font-mono text-[#849588] mt-1">
          &gt; {overPct}% FILL THRESHOLD
        </div>
      </div>

      {/* 5. Mean Fill Ratio */}
      <div className="p-3 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-col justify-between shadow-sm">
        <div className="flex items-center justify-between text-[#849588] text-[10px] font-mono font-bold tracking-wider">
          <span>MEAN LIQUID RATIO</span>
          <Droplets size={14} className="text-[#00daf3]" />
        </div>
        <div className="text-2xl font-mono font-black text-[#00daf3] mt-1 tracking-tight">
          {(meanFillRatio * 100).toFixed(1)}%
        </div>
        <div className="text-[10px] font-mono text-[#849588] mt-1">
          NOMINAL BAND: {Math.round(underPct)}% - {Math.round(overPct)}%
        </div>
      </div>

      {/* 6. Cycle Time */}
      <div className="p-3 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-col justify-between shadow-sm">
        <div className="flex items-center justify-between text-[#849588] text-[10px] font-mono font-bold tracking-wider">
          <span>CYCLE LATENCY</span>
          <Clock size={14} className="text-[#38bdf8]" />
        </div>
        <div className="text-2xl font-mono font-black text-[#38bdf8] mt-1 tracking-tight">
          {typeof latency === 'number' ? latency.toFixed(2) : latency} <span className="text-xs font-normal text-[#849588]">ms</span>
        </div>
        <div className="text-[10px] font-mono text-[#00f5a0] mt-1 flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-[#00f5a0]" />
          <span>{typeof latency === 'number' && latency > 0 ? `${(1000 / latency).toFixed(1)} FPS EQUIV.` : 'AWAITING FIRST CYCLE'}</span>
        </div>
      </div>
    </div>
  );
}
