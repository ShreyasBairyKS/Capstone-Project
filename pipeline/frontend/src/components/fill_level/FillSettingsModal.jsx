import React, { useState, useEffect } from 'react';
import { X, Sliders, CheckCircle2 } from 'lucide-react';
import { useStation } from '../../context/StationContext';

const FIELDS = [
  {
    key: 'underfill_threshold', label: 'Underfill Reject Threshold', min: 0.10, max: 0.60, step: 0.01, color: 'var(--reject-color)',
    help: 'Bottles filled below this fraction of their height are rejected and routed to the Refill station.'
  },
  {
    key: 'overfill_threshold', label: 'Overfill Reject Threshold', min: 0.60, max: 0.98, step: 0.01, color: 'var(--reject-color)',
    help: 'Bottles filled above this fraction of their height are rejected and routed to the Drain station.'
  },
  {
    key: 'borderline_band', label: 'Gray-Zone Band (+/-)', min: 0.0, max: 0.05, step: 0.005, color: 'var(--cyan-color)',
    help: 'Readings this close to either threshold with weak meniscus confidence go to UNCERTAIN for operator review.'
  },
  {
    key: 'borderline_conf_floor', label: 'Meniscus Confidence Floor', min: 0.20, max: 0.80, step: 0.01, color: 'var(--cyan-color)',
    help: 'Gray-zone readings with meniscus confidence below this are routed to UNCERTAIN instead of PASS/REJECT.'
  },
  {
    key: 'bottle_conf_threshold', label: 'Bottle Detector Confidence', min: 0.05, max: 0.95, step: 0.01, color: 'var(--cyan-color)',
    help: 'Minimum confidence for the bottle localizer. Lower finds more bottles but risks false detections.'
  },
  {
    key: 'water_conf_threshold', label: 'Meniscus Detector Confidence', min: 0.05, max: 0.95, step: 0.01, color: 'var(--cyan-color)',
    help: 'Minimum confidence for the liquid-surface detector.'
  },
];

export default function FillSettingsModal({ isOpen, onClose }) {
  const { refreshHealth } = useStation();
  const [config, setConfig] = useState(null);
  const [saving, setSaving] = useState(false);
  const [savedSuccess, setSavedSuccess] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!isOpen) return;
    setError('');
    fetch('/api/fill/config')
      .then(res => res.json())
      .then(data => setConfig(data))
      .catch(err => setError(`Could not load Station 2 settings: ${err.message}`));
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSave = async () => {
    setSaving(true);
    setError('');
    try {
      const res = await fetch('/api/fill/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(typeof body.detail === 'string' ? body.detail : 'Settings rejected by server.');
      }
      setConfig(body);
      refreshHealth();
      setSavedSuccess(true);
      setTimeout(() => setSavedSuccess(false), 2000);
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const pct = v => (v * 100).toFixed(v < 0.1 ? 1 : 0);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" style={{ maxWidth: '600px' }} onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Sliders className="text-cyan-400" size={22} />
            <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem' }}>Station 2 &bull; Fill Level Settings</h2>
          </div>
          <button className="btn btn-secondary" onClick={onClose} style={{ padding: '0.4rem' }}>
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {!config && !error && <span style={{ color: 'var(--text-dim)' }}>Loading...</span>}

          {config && FIELDS.map(f => (
            <div className="slider-group" key={f.key}>
              <div className="slider-label">
                <span>{f.label}</span>
                <strong style={{ color: f.color }}>{pct(config[f.key])}%</strong>
              </div>
              <input
                type="range"
                min={f.min}
                max={f.max}
                step={f.step}
                value={config[f.key]}
                onChange={e => setConfig({ ...config, [f.key]: parseFloat(e.target.value) })}
              />
              <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>{f.help}</span>
            </div>
          ))}

          {error && (
            <div style={{ color: 'var(--reject-color)', fontSize: '0.85rem', marginTop: '0.5rem' }}>{error}</div>
          )}

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1rem' }}>
            <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button className="btn btn-primary" onClick={handleSave} disabled={saving || !config}>
              {savedSuccess ? (
                <>
                  <CheckCircle2 size={16} />
                  <span>Thresholds Applied!</span>
                </>
              ) : (
                <span>{saving ? 'Applying...' : 'Apply to Station 2'}</span>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
