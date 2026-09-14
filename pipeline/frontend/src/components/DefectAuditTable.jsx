import React from 'react';
import { AlertTriangle, Eye, ShieldAlert, CheckCircle2 } from 'lucide-react';

export default function DefectAuditTable({ defects, onSelectBottle }) {
  return (
    <div className="glass-card audit-table-card">
      <div className="panel-header">
        <div className="panel-title">
          <AlertTriangle className="text-amber-400" size={20} />
          <span>Defect Audit & Active Learning Log</span>
        </div>
        <span className="chip" style={{ fontFamily: 'var(--font-mono)' }}>
          {defects.length} Recorded Defect Events
        </span>
      </div>

      <div className="table-responsive">
        <table>
          <thead>
            <tr>
              <th>BOTTLE ID</th>
              <th>TIMESTAMP</th>
              <th>PRIMARY DEFECT</th>
              <th>CONFIDENCE</th>
              <th>FLAGGED VIEWS</th>
              <th>ACTUATOR</th>
              <th>CYCLE TIME</th>
              <th>ACTIONS</th>
            </tr>
          </thead>
          <tbody>
            {defects.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-dim)' }}>
                  No defects logged yet. Press "Simulate Bottle" to generate inspection events.
                </td>
              </tr>
            ) : (
              defects.map((d) => (
                <tr key={d.bottle_id} onClick={() => onSelectBottle(d.bottle_id)}>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#38bdf8' }}>
                    {d.bottle_id}
                  </td>
                  <td style={{ color: 'var(--text-muted)' }}>
                    {new Date(d.timestamp).toLocaleTimeString()}
                  </td>
                  <td>
                    <span className="chip" style={{ background: 'rgba(244, 63, 94, 0.1)', color: '#f43f5e', borderColor: 'rgba(244, 63, 94, 0.3)' }}>
                      {d.primary_defect || 'Defect'}
                    </span>
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {d.primary_defect_confidence ? `${(d.primary_defect_confidence * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td>
                    {d.flagged_cameras && d.flagged_cameras.length > 0 
                      ? d.flagged_cameras.join(', ')
                      : 'All views'}
                  </td>
                  <td>
                    {d.trigger_actuator ? (
                      <span style={{ color: '#f43f5e', fontWeight: 600 }}>TRIGGERED</span>
                    ) : (
                      <span style={{ color: '#10b981', fontWeight: 600 }}>STANDBY</span>
                    )}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {d.total_latency_ms ? `${d.total_latency_ms.toFixed(1)} ms` : '—'}
                  </td>
                  <td>
                    <button 
                      className="btn btn-secondary" 
                      style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectBottle(d.bottle_id);
                      }}
                    >
                      <Eye size={12} />
                      <span>Review Replay</span>
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
