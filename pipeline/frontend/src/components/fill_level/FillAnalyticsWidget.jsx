import React, { useEffect, useState } from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell, 
  PieChart, 
  Pie 
} from 'recharts';
import { BarChart3, PieChart as PieIcon } from 'lucide-react';
import { useStation } from '../../context/StationContext';

export default function FillAnalyticsWidget({ stats, defects = [] }) {
  const { fillHealth } = useStation();
  const underThresh = fillHealth?.underfill_threshold ?? 0.40;
  const overThresh = fillHealth?.overfill_threshold ?? 0.85;
  const [histogram, setHistogram] = useState([]);

  // Real measured fill-ratio distribution (10% bins) from the Station 2 audit DB; refreshed as inspections arrive
  const totalInspected = stats?.total_inspected ?? 0;
  useEffect(() => {
    let cancelled = false;
    fetch('/api/fill/ratio-histogram?bins=10')
      .then(r => (r.ok ? r.json() : []))
      .then(rows => { if (!cancelled && Array.isArray(rows)) setHistogram(rows); })
      .catch(() => null);
    return () => { cancelled = true; };
  }, [totalInspected]);

  const passCount = stats?.pass_count ?? 0;
  const underfillCount = stats?.underfill_count ?? 0;
  const overfillCount = stats?.overfill_count ?? 0;
  const dryCount = stats?.dry_count ?? 0;
  const total = stats?.total_inspected ?? (passCount + underfillCount + overfillCount + dryCount);

  // Category chart data
  const categoryData = [
    { name: `Nominal (${Math.round(underThresh * 100)}-${Math.round(overThresh * 100)}%)`, count: passCount, color: '#00f5a0' },
    { name: `Underfill (<${Math.round(underThresh * 100)}%)`, count: underfillCount, color: '#f59e0b' },
    { name: `Overfill (>${Math.round(overThresh * 100)}%)`, count: overfillCount, color: '#ff3b30' },
    { name: 'Dry / No Liquid', count: dryCount, color: '#94a3b8' }
  ];

  // Color each bin by where its center falls relative to the configured nominal band
  const ratioBins = histogram.map(bin => {
    const center = (bin.lo + bin.hi) / 2;
    const color = center < underThresh ? '#f59e0b' : center > overThresh ? '#ff3b30' : '#00f5a0';
    return { range: `${Math.round(bin.lo * 100)}-${Math.round(bin.hi * 100)}%`, count: bin.count, color };
  });

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5">
      {/* 1. Volume Histogram (8 Cols) */}
      <div className="lg:col-span-8 p-3.5 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-col gap-2.5 shadow-sm">
        <div className="flex items-center justify-between border-b border-[#1e2e48] pb-2">
          <div className="flex items-center gap-2 text-xs font-mono font-bold text-[#dae2fd]">
            <BarChart3 size={15} className="text-[#00daf3]" />
            <span>LIQUID FILL RATIO DISTRIBUTION (TOLERANCE {Math.round(underThresh * 100)}% - {Math.round(overThresh * 100)}%)</span>
          </div>
          <div className="text-[10px] font-mono text-[#849588]">
            MEASURED BOTTLES: {ratioBins.reduce((n, b) => n + b.count, 0)}
          </div>
        </div>

        <div className="h-56 w-full pt-1">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={ratioBins} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2e48" vertical={false} />
              <XAxis dataKey="range" stroke="#849588" fontSize={11} fontFamily="monospace" />
              <YAxis stroke="#849588" fontSize={11} fontFamily="monospace" allowDecimals={false} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#060a14', 
                  borderColor: '#1e2e48', 
                  borderRadius: '6px',
                  fontFamily: 'monospace',
                  fontSize: '11px',
                  color: '#dae2fd'
                }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {ratioBins.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="flex items-center justify-between pt-1 border-t border-[#1e2e48] text-[10px] font-mono text-[#849588]">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-[#f59e0b]" /> UNDERFILL</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-[#00f5a0]" /> NOMINAL TARGET</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-[#ff3b30]" /> OVERFILL</span>
          </div>
          <span>TOTAL SPECIMENS: {total}</span>
        </div>
      </div>

      {/* 2. Fill Verdict Breakdown (4 Cols) */}
      <div className="lg:col-span-4 p-3.5 rounded bg-[#0b1326] border border-[#1e2e48] flex flex-col justify-between gap-2.5 shadow-sm">
        <div className="flex items-center justify-between border-b border-[#1e2e48] pb-2">
          <div className="flex items-center gap-2 text-xs font-mono font-bold text-[#dae2fd]">
            <PieIcon size={15} className="text-[#00f5a0]" />
            <span>QUALITY YIELD BREAKDOWN</span>
          </div>
          <span className="text-[10px] font-mono text-[#00f5a0]">STATION 2</span>
        </div>

        <div className="flex flex-col gap-2 my-auto">
          {categoryData.map((cat, idx) => {
            const pct = total > 0 ? ((cat.count / total) * 100).toFixed(1) : '0.0';
            return (
              <div key={idx} className="flex flex-col gap-1 p-2 rounded bg-[#070d1c] border border-[#1e2e48]">
                <div className="flex items-center justify-between text-xs font-mono">
                  <div className="flex items-center gap-1.5 font-bold text-[#dae2fd]">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: cat.color }} />
                    <span>{cat.name}</span>
                  </div>
                  <span className="font-bold text-[#dae2fd]">{cat.count} ({pct}%)</span>
                </div>
                <div className="w-full h-1.5 bg-[#131b2e] rounded overflow-hidden">
                  <div
                    className="h-full rounded"
                    style={{ width: `${pct}%`, backgroundColor: cat.color }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        <div className="pt-2 border-t border-[#1e2e48] text-[10px] font-mono text-[#849588] text-center">
          REAL-TIME TELEMETRY STREAM &bull; SQLITE WAL SYNC
        </div>
      </div>
    </div>
  );
}
