import React, { useState, useEffect } from 'react';
import { X, Eye, Layers, Camera, Download, FileText, CheckCircle2 } from 'lucide-react';

export default function ReplayModal({ bottleId, onClose }) {
  const [detail, setDetail] = useState(null);
  const [selectedCam, setSelectedCam] = useState('camera_front');
  const [renderOverlay, setRenderOverlay] = useState(true);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!bottleId) return;
    setLoading(true);
    fetch(`/api/bottles/${bottleId}`)
      .then(res => res.json())
      .then(data => {
        setDetail(data);
        const images = data?.manifest?.bundle_manifest?.images || {};
        const cams = Object.keys(images);
        if (cams.length > 0) setSelectedCam(cams[0]);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, [bottleId]);

  if (!bottleId) return null;

  const imageUrl = `/api/bottles/${bottleId}/image/${selectedCam}?render=${renderOverlay}&t=${Date.now()}`;
  const manifest = detail?.manifest;
  const record = detail?.record;
  const availableCams = Object.keys(manifest?.bundle_manifest?.images || { camera_front: true });

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Layers className="text-cyan-400" size={24} />
            <div>
              <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem' }}>Dynamic Defect Replay Inspector</h2>
              <span className="chip" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>{bottleId}</span>
            </div>
          </div>
          <button className="btn btn-secondary" onClick={onClose} style={{ padding: '0.4rem' }}>
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {/* Controls Bar */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
            {/* Camera View Selector */}
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {availableCams.map(cam => (
                <button
                  key={cam}
                  className={`btn ${selectedCam === cam ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setSelectedCam(cam)}
                >
                  <Camera size={14} />
                  <span>{cam.toUpperCase()}</span>
                </button>
              ))}
            </div>

            {/* Dynamic Rendering Mode Switcher */}
            <div style={{ display: 'flex', gap: '0.5rem', background: 'rgba(0,0,0,0.3)', padding: '4px', borderRadius: 'var(--radius-md)' }}>
              <button
                className={`btn ${!renderOverlay ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
                onClick={() => setRenderOverlay(false)}
              >
                Raw Clean Image (Training)
              </button>
              <button
                className={`btn ${renderOverlay ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
                onClick={() => setRenderOverlay(true)}
              >
                Dynamic Transparent Overlay
              </button>
            </div>
          </div>

          {/* Image Display */}
          <div style={{ 
            background: '#000', 
            borderRadius: 'var(--radius-md)', 
            overflow: 'hidden', 
            minHeight: '400px', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            position: 'relative' 
          }}>
            {loading ? (
              <span style={{ color: 'var(--text-dim)' }}>Loading inspection bundle...</span>
            ) : (
              <img 
                src={imageUrl} 
                alt={`${bottleId} ${selectedCam}`} 
                style={{ maxWidth: '100%', maxHeight: '600px', objectFit: 'contain' }} 
              />
            )}
            <div style={{ position: 'absolute', bottom: '12px', right: '12px' }}>
              <span className="chip" style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)' }}>
                {renderOverlay ? 'Dynamic Overlay Active' : 'Pristine 12MP Pixels'}
              </span>
            </div>
          </div>

          {/* Metadata Card */}
          {record && (
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
              gap: '1rem', 
              background: 'rgba(255,255,255,0.03)', 
              padding: '1rem', 
              borderRadius: 'var(--radius-md)' 
            }}>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>VERDICT</div>
                <div style={{ fontWeight: 700, color: record.is_compliant ? '#10b981' : '#f43f5e' }}>
                  {record.global_verdict}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>PRIMARY DEFECT</div>
                <div style={{ fontWeight: 600 }}>{record.primary_defect || 'None'}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>CONFIDENCE</div>
                <div style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                  {record.primary_defect_confidence ? `${(record.primary_defect_confidence * 100).toFixed(1)}%` : '—'}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>ACTIVE LEARNING EXPORT</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: '#38bdf8', fontSize: '0.85rem' }}>
                  <FileText size={14} />
                  <span>YOLO .txt generated</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
