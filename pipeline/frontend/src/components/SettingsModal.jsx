import React, { useState, useEffect } from 'react';
import { X, Sliders, CheckCircle2 } from 'lucide-react';

export default function SettingsModal({ isOpen, onClose }) {
  const [config, setConfig] = useState({
    pass_conf_threshold: 0.75,
    defect_conf_threshold: 0.45,
    uncertain_floor: 0.40,
    mask_alpha: 0.40,
    save_pass_images: true
  });
  const [saving, setSaving] = useState(false);
  const [savedSuccess, setSavedSuccess] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    fetch('/api/config')
      .then(res => res.json())
      .then(data => setConfig(data))
      .catch(err => console.error(err));
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSave = async () => {
    setSaving(true);
    try {
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      if (res.ok) {
        setSavedSuccess(true);
        setTimeout(() => setSavedSuccess(false), 2000);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" style={{ maxWidth: '600px' }} onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Sliders className="text-cyan-400" size={22} />
            <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem' }}>Quality Gate Threshold Settings</h2>
          </div>
          <button className="btn btn-secondary" onClick={onClose} style={{ padding: '0.4rem' }}>
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {/* PASS Threshold Slider */}
          <div className="slider-group">
            <div className="slider-label">
              <span>Good Cap PASS Confidence Threshold</span>
              <strong style={{ color: 'var(--pass-color)' }}>{(config.pass_conf_threshold * 100).toFixed(0)}%</strong>
            </div>
            <input 
              type="range" 
              min="0.50" 
              max="0.95" 
              step="0.01" 
              value={config.pass_conf_threshold}
              onChange={e => setConfig({ ...config, pass_conf_threshold: parseFloat(e.target.value) })}
            />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              Caps must exceed this confidence across all views to earn PASS certification.
            </span>
          </div>

          {/* REJECT Threshold Slider */}
          <div className="slider-group">
            <div className="slider-label">
              <span>Defect REJECT Trigger Threshold</span>
              <strong style={{ color: 'var(--reject-color)' }}>{(config.defect_conf_threshold * 100).toFixed(0)}%</strong>
            </div>
            <input 
              type="range" 
              min="0.25" 
              max="0.80" 
              step="0.01" 
              value={config.defect_conf_threshold}
              onChange={e => setConfig({ ...config, defect_conf_threshold: parseFloat(e.target.value) })}
            />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              Any camera angle observing a defect at or above this confidence immediately triggers ejection.
            </span>
          </div>

          {/* Mask Alpha Slider */}
          <div className="slider-group">
            <div className="slider-label">
              <span>Visual Overlay Mask Transparency (Alpha)</span>
              <strong style={{ color: 'var(--cyan-color)' }}>{(config.mask_alpha * 100).toFixed(0)}%</strong>
            </div>
            <input 
              type="range" 
              min="0.10" 
              max="0.90" 
              step="0.05" 
              value={config.mask_alpha}
              onChange={e => setConfig({ ...config, mask_alpha: parseFloat(e.target.value) })}
            />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              Controls the opacity blend of segmentation masks over the raw cap texture.
            </span>
          </div>

          {/* Save Button */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1rem' }}>
            <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
              {savedSuccess ? (
                <>
                  <CheckCircle2 size={16} />
                  <span>Thresholds Applied!</span>
                </>
              ) : (
                <span>{saving ? 'Applying...' : 'Apply to Pipeline'}</span>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
