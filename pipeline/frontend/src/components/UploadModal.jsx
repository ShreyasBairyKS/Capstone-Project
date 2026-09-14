import React, { useState, useRef } from 'react';
import { X, UploadCloud, Image as ImageIcon, Camera, CheckCircle2, ShieldAlert, ArrowRight } from 'lucide-react';

export default function UploadModal({ isOpen, onClose, onInspectionComplete }) {
  const [frontFile, setFrontFile] = useState(null);
  const [rearFile, setRearFile] = useState(null);
  const [frontPreview, setFrontPreview] = useState(null);
  const [rearPreview, setRearPreview] = useState(null);
  const [customBottleId, setCustomBottleId] = useState('');
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const frontInputRef = useRef(null);
  const rearInputRef = useRef(null);

  if (!isOpen) return null;

  const handleFileSelect = (file, isFront) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      if (isFront) {
        setFrontFile(file);
        setFrontPreview(e.target.result);
      } else {
        setRearFile(file);
        setRearPreview(e.target.result);
      }
    };
    reader.readAsDataURL(file);
    setErrorMsg('');
  };

  const handleDrop = (e, isFront) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0], isFront);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!frontFile && !rearFile) {
      setErrorMsg('Please select at least one bottle image to inspect.');
      return;
    }

    setLoading(true);
    setErrorMsg('');

    try {
      const formData = new FormData();
      const cameraIds = [];

      if (frontFile) {
        formData.append('files', frontFile);
        cameraIds.push('camera_front');
      }
      if (rearFile) {
        formData.append('files', rearFile);
        cameraIds.push('camera_rear');
      }

      formData.append('camera_ids', cameraIds.join(','));
      if (customBottleId.trim()) {
        formData.append('bottle_id', customBottleId.trim());
      }

      const res = await fetch('/api/inspect/image', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Inspection failed');
      }

      const inspectResult = await res.json();
      onInspectionComplete(inspectResult);
      handleReset();
      onClose();
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || 'Error executing inspection');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFrontFile(null);
    setRearFile(null);
    setFrontPreview(null);
    setRearPreview(null);
    setCustomBottleId('');
    setErrorMsg('');
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" style={{ maxWidth: '780px' }} onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <UploadCloud className="text-cyan-400" size={24} />
            <div>
              <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem' }}>
                Manual Bottle Image Inspection
              </h2>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Select image files directly from your computer to run model inference and defect evaluation.
              </p>
            </div>
          </div>
          <button className="btn btn-secondary" onClick={onClose} style={{ padding: '0.4rem' }}>
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="modal-body">
          {errorMsg && (
            <div style={{ 
              background: 'rgba(244, 63, 94, 0.15)', 
              border: '1px solid rgba(244, 63, 94, 0.4)', 
              color: '#fb7185', 
              padding: '0.75rem 1rem', 
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.85rem'
            }}>
              {errorMsg}
            </div>
          )}

          {/* Optional Bottle ID Input */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Custom Bottle ID (Optional)
            </label>
            <input
              type="text"
              placeholder="e.g. BOTTLE_BATCH_042 (Auto-generated if empty)"
              value={customBottleId}
              onChange={e => setCustomBottleId(e.target.value)}
              style={{
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.6rem 0.85rem',
                color: '#fff',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.85rem'
              }}
            />
          </div>

          {/* Camera File Slots */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
            {/* Front Camera Slot */}
            <div 
              style={{
                border: '2px dashed var(--border-highlight)',
                borderRadius: 'var(--radius-md)',
                padding: '1.25rem',
                textAlign: 'center',
                background: frontPreview ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.02)',
                cursor: 'pointer',
                position: 'relative',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '220px',
                transition: 'border-color 0.2s'
              }}
              onClick={() => frontInputRef.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => handleDrop(e, true)}
            >
              <input 
                ref={frontInputRef}
                type="file" 
                accept="image/*" 
                style={{ display: 'none' }}
                onChange={e => e.target.files && handleFileSelect(e.target.files[0], true)}
              />

              {frontPreview ? (
                <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                  <img 
                    src={frontPreview} 
                    alt="Front view" 
                    style={{ maxHeight: '180px', maxWidth: '100%', borderRadius: 'var(--radius-sm)', objectFit: 'contain' }} 
                  />
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#38bdf8', fontWeight: 600 }}>
                    Front: {frontFile?.name}
                  </div>
                  <button
                    type="button"
                    style={{
                      position: 'absolute',
                      top: '-6px',
                      right: '-6px',
                      background: '#f43f5e',
                      color: '#fff',
                      border: 'none',
                      borderRadius: '50%',
                      width: '24px',
                      height: '24px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      setFrontFile(null);
                      setFrontPreview(null);
                    }}
                  >
                    <X size={14} />
                  </button>
                </div>
              ) : (
                <>
                  <Camera size={36} className="text-cyan-400" style={{ marginBottom: '0.75rem', opacity: 0.8 }} />
                  <div style={{ fontWeight: 600, fontSize: '0.95rem', color: 'var(--text-main)' }}>
                    Select Front Camera Image
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: '0.25rem' }}>
                    Click to browse or drag image here
                  </div>
                </>
              )}
            </div>

            {/* Rear Camera Slot */}
            <div 
              style={{
                border: '2px dashed var(--border-highlight)',
                borderRadius: 'var(--radius-md)',
                padding: '1.25rem',
                textAlign: 'center',
                background: rearPreview ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.02)',
                cursor: 'pointer',
                position: 'relative',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '220px',
                transition: 'border-color 0.2s'
              }}
              onClick={() => rearInputRef.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => handleDrop(e, false)}
            >
              <input 
                ref={rearInputRef}
                type="file" 
                accept="image/*" 
                style={{ display: 'none' }}
                onChange={e => e.target.files && handleFileSelect(e.target.files[0], false)}
              />

              {rearPreview ? (
                <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                  <img 
                    src={rearPreview} 
                    alt="Rear view" 
                    style={{ maxHeight: '180px', maxWidth: '100%', borderRadius: 'var(--radius-sm)', objectFit: 'contain' }} 
                  />
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#38bdf8', fontWeight: 600 }}>
                    Rear: {rearFile?.name}
                  </div>
                  <button
                    type="button"
                    style={{
                      position: 'absolute',
                      top: '-6px',
                      right: '-6px',
                      background: '#f43f5e',
                      color: '#fff',
                      border: 'none',
                      borderRadius: '50%',
                      width: '24px',
                      height: '24px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      setRearFile(null);
                      setRearPreview(null);
                    }}
                  >
                    <X size={14} />
                  </button>
                </div>
              ) : (
                <>
                  <Camera size={36} className="text-cyan-400" style={{ marginBottom: '0.75rem', opacity: 0.8 }} />
                  <div style={{ fontWeight: 600, fontSize: '0.95rem', color: 'var(--text-main)' }}>
                    Select Rear Camera Image (Optional)
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: '0.25rem' }}>
                    Click to browse or drag image here
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '0.5rem' }}>
            <button type="button" className="btn btn-secondary" onClick={onClose} disabled={loading}>
              Cancel
            </button>
            <button 
              type="submit" 
              className="btn btn-primary" 
              disabled={loading || (!frontFile && !rearFile)}
            >
              <UploadCloud size={16} />
              <span>{loading ? 'Processing Model Inference...' : 'Run Inspection'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
