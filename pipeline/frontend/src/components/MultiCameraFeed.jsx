import React from 'react';
import { Camera, ShieldAlert, ShieldCheck, AlertTriangle } from 'lucide-react';

export default function MultiCameraFeed({ latestInspection }) {
  const views = latestInspection?.views || {};
  const globalVerdict = latestInspection?.global_verdict || 'PASS';
  const isCompliant = latestInspection?.is_compliant ?? true;
  const primaryDefect = latestInspection?.primary_defect;
  const decisionReason = latestInspection?.decision_reason || 'Conveyor ready. Awaiting bottle inspection.';
  const bottleId = latestInspection?.bottle_id || 'STANDBY_MODE';

  const cameraKeys = Object.keys(views);
  const displayCams = cameraKeys.length > 0 ? cameraKeys : ['camera_front', 'camera_rear'];

  return (
    <div className="glass-card inspection-panel">
      {/* Global Decision Banner */}
      <div className={`global-inspection-card ${globalVerdict}`}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {globalVerdict === 'PASS' && <ShieldCheck size={32} color="#10b981" />}
          {globalVerdict === 'REJECT' && <ShieldAlert size={32} color="#f43f5e" />}
          {globalVerdict === 'UNCERTAIN' && <AlertTriangle size={32} color="#f59e0b" />}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.4rem', fontWeight: 700 }}>
                {globalVerdict === 'REJECT' ? `EJECT: ${primaryDefect?.toUpperCase() || 'DEFECT'}` : globalVerdict}
              </h2>
              <span className="chip" style={{ fontFamily: 'var(--font-mono)' }}>{bottleId}</span>
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '4px' }}>
              {decisionReason}
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '1.5rem', textAlign: 'right' }}>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>ACTUATOR STATUS</div>
            <div style={{ fontWeight: 700, color: latestInspection?.trigger_reject_actuator ? '#f43f5e' : '#10b981' }}>
              {latestInspection?.trigger_reject_actuator ? 'EJECT SIGNAL SENT' : 'CONVEYOR PASS'}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>INSPECTION LATENCY</div>
            <div style={{ fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
              {latestInspection?.total_latency_ms ? `${latestInspection.total_latency_ms} ms` : '—'}
            </div>
          </div>
        </div>
      </div>

      {/* Camera Grid Viewports */}
      <div className="camera-grid">
        {displayCams.map((camId) => {
          const viewData = views[camId];
          const hasImage = !!viewData?.annotated_image_base64;
          const verdict = viewData?.verdict || 'STANDBY';
          const defectDetected = viewData?.defect_detected;
          const conf = viewData?.primary_confidence;

          return (
            <div key={camId} className="camera-viewport">
              <div className="camera-viewport-header">
                <div className="cam-name">
                  <Camera size={16} className="text-cyan-400" />
                  <span>{camId.toUpperCase()}</span>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <span className="chip" style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem' }}>
                    {viewData?.detection_count !== undefined ? `${viewData.detection_count} obj` : '1080p 25fps'}
                  </span>
                </div>
              </div>

              <div className="viewport-canvas-container">
                {hasImage ? (
                  <>
                    <img 
                      src={viewData.annotated_image_base64} 
                      alt={camId} 
                      className="viewport-img" 
                    />
                    <div className={`view-verdict-overlay ${verdict}`}>
                      {verdict === 'REJECT' ? `REJECT: ${viewData?.defect_name || 'DEFECT'}` : verdict}
                      {conf ? ` (${(conf * 100).toFixed(1)}%)` : ''}
                    </div>
                  </>
                ) : (
                  <div className="viewport-empty-placeholder">
                    <Camera size={36} style={{ opacity: 0.3 }} />
                    <span>Awaiting Inspection Trigger...</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
