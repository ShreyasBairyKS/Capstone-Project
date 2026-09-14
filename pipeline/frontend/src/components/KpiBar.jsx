import React from 'react';
import { CheckCircle2, XCircle, Gauge, Layers } from 'lucide-react';

export default function KpiBar({ stats, latestLatency }) {
  const total = stats?.total_inspected || 0;
  const passed = stats?.passed || 0;
  const rejected = stats?.rejected || 0;
  const passRate = stats?.pass_rate_pct ?? 100.0;
  const rejectRate = stats?.reject_rate_pct ?? 0.0;
  const avgLatency = stats?.average_latency_ms || 0;
  const defectBreakdown = stats?.defect_breakdown || {};

  return (
    <div className="kpi-grid">
      {/* Total Inspected */}
      <div className="glass-card kpi-card">
        <div className="kpi-header">
          <span>TOTAL INSPECTED</span>
          <Layers size={18} className="text-cyan-400" />
        </div>
        <div className="kpi-value cyan">{total.toLocaleString()}</div>
        <div className="kpi-subtext">Cumulative inspection cycles</div>
      </div>

      {/* Compliance Pass Yield */}
      <div className="glass-card kpi-card">
        <div className="kpi-header">
          <span>COMPLIANCE YIELD</span>
          <CheckCircle2 size={18} className="text-emerald-400" />
        </div>
        <div className="kpi-value pass">{passRate}%</div>
        <div className="progress-bar-bg">
          <div className="progress-bar-fill pass" style={{ width: `${passRate}%` }} />
        </div>
        <div className="kpi-subtext">{passed} certified compliant</div>
      </div>

      {/* Rejection Rate */}
      <div className="glass-card kpi-card">
        <div className="kpi-header">
          <span>REJECTION RATE</span>
          <XCircle size={18} className="text-rose-400" />
        </div>
        <div className="kpi-value reject">{rejectRate}%</div>
        <div className="progress-bar-bg">
          <div className="progress-bar-fill reject" style={{ width: `${rejectRate}%` }} />
        </div>
        <div className="kpi-subtext">{rejected} defective bottles ejected</div>
      </div>

      {/* Pipeline Cycle Latency */}
      <div className="glass-card kpi-card">
        <div className="kpi-header">
          <span>AVG CYCLE TIME</span>
          <Gauge size={18} className="text-cyan-400" />
        </div>
        <div className="kpi-value cyan">
          {latestLatency ? `${latestLatency.toFixed(1)}` : `${avgLatency.toFixed(1)}`}
          <span style={{ fontSize: '1rem', color: '#94a3b8', marginLeft: '4px' }}>ms</span>
        </div>
        <div className="kpi-subtext">Target: &lt; 50ms (GPU FP16)</div>
      </div>
    </div>
  );
}
