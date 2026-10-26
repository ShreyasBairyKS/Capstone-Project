import React, { useState, useRef, useEffect } from 'react';
import { X, UploadCloud, Droplets, Camera, CheckCircle2, AlertTriangle, ShieldCheck, FileText, Loader2 } from 'lucide-react';

export default function FillUploadModal({ isOpen, onClose, onInspectionComplete }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [customBottleId, setCustomBottleId] = useState('');
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);

  // Close on ESC key press
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen && !loading) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, loading, onClose]);

  if (!isOpen) return null;

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith('image/') && !selectedFile.name.match(/\.(png|jpe?g|bmp|webp)$/i)) {
      setErrorMsg('Please select a valid image file (PNG, JPG, BMP, WebP).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      setFile(selectedFile);
      setPreview(e.target.result);
      setErrorMsg('');
    };
    reader.onerror = () => {
      setErrorMsg('Failed to read image file.');
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setCustomBottleId('');
    setErrorMsg('');
    setIsDragging(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setErrorMsg('Please select a bottle profile image to inspect.');
      return;
    }

    setLoading(true);
    setErrorMsg('');

    try {
      const formData = new FormData();
      // Station 2 takes exactly 1 file parameter: 'file'
      formData.append('file', file);
      if (customBottleId.trim()) {
        formData.append('bottle_id', customBottleId.trim());
      }

      // Station 2 dedicated endpoint - Station 1 endpoints are never called
      const url = customBottleId.trim()
        ? `/api/fill/inspect?bottle_id=${encodeURIComponent(customBottleId.trim())}`
        : '/api/fill/inspect';

      const res = await fetch(url, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Station 2 inspection failed with status ${res.status}`);
      }

      const inspectResult = await res.json();
      onInspectionComplete(inspectResult);
      handleReset();
      onClose();
    } catch (err) {
      console.error('Station 2 upload error:', err);
      setErrorMsg(err.message || 'Error executing fill level inspection');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={loading ? undefined : onClose}>
      <div 
        className="modal-card" 
        style={{ maxWidth: '780px' }} 
        onClick={e => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div 
              style={{ 
                background: 'rgba(0, 218, 243, 0.12)', 
                border: '1px solid rgba(0, 218, 243, 0.4)', 
                borderRadius: 'var(--radius-sm)',
                padding: '0.5rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#00daf3'
              }}
            >
              <Droplets size={24} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', fontWeight: 600, color: '#dae2fd' }}>
                  Bottle Profile Liquid Inspection
                </h2>
                <span 
                  style={{
                    fontSize: '0.65rem',
                    fontFamily: 'var(--font-mono)',
                    padding: '0.15rem 0.45rem',
                    borderRadius: '4px',
                    background: 'rgba(0, 218, 243, 0.1)',
                    border: '1px solid rgba(0, 218, 243, 0.3)',
                    color: '#00daf3',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em'
                  }}
                >
                  Station 2 &bull; CAM-04
                </span>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                Upload a single side-profile bottle specimen image to detect liquid meniscus, compute fill volume ratio, and evaluate fill tolerance compliance.
              </p>
            </div>
          </div>
          <button 
            type="button"
            className="btn btn-secondary" 
            onClick={onClose} 
            disabled={loading}
            style={{ padding: '0.45rem', cursor: loading ? 'not-allowed' : 'pointer' }}
            title="Close Modal"
          >
            <X size={18} />
          </button>
        </div>

        {/* Modal Body */}
        <form onSubmit={handleSubmit} className="modal-body" style={{ gap: '1.25rem' }}>
          {errorMsg && (
            <div style={{ 
              background: 'rgba(244, 63, 94, 0.15)', 
              border: '1px solid rgba(244, 63, 94, 0.4)', 
              color: '#fb7185', 
              padding: '0.75rem 1rem', 
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              <AlertTriangle size={16} className="shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Optional Bottle ID Input */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <label style={{ fontSize: '0.825rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', display: 'flex', justifyContent: 'space-between' }}>
              <span>BOTTLE SPECIMEN ID (OPTIONAL)</span>
              <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Leave blank for auto-generated ID</span>
            </label>
            <input
              type="text"
              placeholder="e.g. FILL_SPECIMEN_042"
              value={customBottleId}
              onChange={e => setCustomBottleId(e.target.value)}
              disabled={loading}
              style={{
                background: 'rgba(0, 0, 0, 0.35)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.6rem 0.85rem',
                color: '#fff',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.85rem',
                outline: 'none',
                transition: 'border-color 0.2s'
              }}
              onFocus={e => e.target.style.borderColor = '#00daf3'}
              onBlur={e => e.target.style.borderColor = 'var(--border-subtle)'}
            />
          </div>

          {/* Dedicated Single Profile Bottle Dropzone */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <label style={{ fontSize: '0.825rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              PROFILE IMAGE SPECIMEN (EXACTLY 1 IMAGE)
            </label>

            <div
              style={{
                border: isDragging 
                  ? '2px dashed #00daf3' 
                  : file 
                    ? '2px solid rgba(0, 245, 160, 0.5)' 
                    : '2px dashed var(--border-highlight)',
                borderRadius: 'var(--radius-md)',
                padding: '1.5rem',
                textAlign: 'center',
                background: isDragging 
                  ? 'rgba(0, 218, 243, 0.08)' 
                  : preview 
                    ? 'rgba(0, 0, 0, 0.45)' 
                    : 'rgba(255, 255, 255, 0.02)',
                cursor: loading ? 'default' : 'pointer',
                position: 'relative',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '260px',
                transition: 'all 0.2s ease',
                boxShadow: file ? '0 0 15px rgba(0, 245, 160, 0.1)' : 'none'
              }}
              onClick={() => !loading && fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <input 
                ref={fileInputRef}
                type="file" 
                accept="image/*" 
                style={{ display: 'none' }}
                disabled={loading}
                onChange={e => e.target.files && e.target.files.length > 0 && handleFileSelect(e.target.files[0])}
              />

              {preview ? (
                <div style={{ position: 'relative', width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <img 
                    src={preview} 
                    alt="Bottle Profile Preview" 
                    style={{ 
                      maxHeight: '220px', 
                      maxWidth: '100%', 
                      borderRadius: 'var(--radius-sm)', 
                      objectFit: 'contain',
                      border: '1px solid rgba(255, 255, 255, 0.1)'
                    }} 
                  />

                  <div style={{ marginTop: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'center' }}>
                    <span style={{ fontSize: '0.825rem', color: '#00daf3', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                      {file?.name}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: '#849588', fontFamily: 'var(--font-mono)' }}>
                      ({file ? (file.size / 1024).toFixed(1) : 0} KB)
                    </span>
                    <span style={{ 
                      fontSize: '0.7rem', 
                      fontFamily: 'var(--font-mono)', 
                      color: '#00f5a0', 
                      background: 'rgba(0, 245, 160, 0.1)', 
                      padding: '0.15rem 0.4rem', 
                      borderRadius: '3px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.25rem'
                    }}>
                      <CheckCircle2 size={11} />
                      READY FOR S2 INFERENCE
                    </span>
                  </div>

                  {!loading && (
                    <button
                      type="button"
                      title="Remove image"
                      style={{
                        position: 'absolute',
                        top: '-8px',
                        right: '0px',
                        background: '#f43f5e',
                        color: '#fff',
                        border: 'none',
                        borderRadius: '50%',
                        width: '26px',
                        height: '26px',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        boxShadow: '0 2px 8px rgba(244, 63, 94, 0.5)'
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleReset();
                      }}
                    >
                      <X size={14} />
                    </button>
                  )}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem', pointerEvents: 'none' }}>
                  <div 
                    style={{ 
                      width: '56px', 
                      height: '56px', 
                      borderRadius: '50%', 
                      background: 'rgba(0, 218, 243, 0.08)', 
                      border: '1px solid rgba(0, 218, 243, 0.25)', 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      marginBottom: '0.25rem'
                    }}
                  >
                    <Droplets size={28} className="text-cyan-400" />
                  </div>
                  <div style={{ fontWeight: 600, fontSize: '1rem', color: 'var(--text-main)', fontFamily: 'var(--font-sans)' }}>
                    Select Single Profile Bottle Image
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', maxWidth: '340px' }}>
                    Click to browse or drag and drop a bottle profile specimen here.
                  </div>
                  <div style={{ 
                    fontSize: '0.72rem', 
                    fontFamily: 'var(--font-mono)', 
                    color: '#849588', 
                    marginTop: '0.4rem',
                    background: 'rgba(0, 0, 0, 0.3)',
                    padding: '0.3rem 0.65rem',
                    borderRadius: '4px',
                    border: '1px solid var(--border-subtle)'
                  }}>
                    CALIBRATED PROFILE SENSOR &bull; MONOCHROME / COLOR &bull; 640&times;640
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Machine Vision Inspection Guidance Banner */}
          <div 
            style={{
              padding: '0.85rem 1rem',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(11, 19, 38, 0.7)',
              border: '1px solid #1e2e48',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '0.75rem',
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono)'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <div style={{ color: '#00daf3', marginTop: '1px' }}>1.</div>
              <div>
                <span style={{ color: '#dae2fd', fontWeight: 600 }}>Profile Localization:</span>
                <p style={{ color: '#849588', marginTop: '2px', fontSize: '0.7rem' }}>
                  Detects container geometry and vertical height bounds.
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <div style={{ color: '#00daf3', marginTop: '1px' }}>2.</div>
              <div>
                <span style={{ color: '#dae2fd', fontWeight: 600 }}>Meniscus Detection:</span>
                <p style={{ color: '#849588', marginTop: '2px', fontSize: '0.7rem' }}>
                  Locates liquid surface line &amp; calculates fill volume ratio.
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <div style={{ color: '#00f5a0', marginTop: '1px' }}>3.</div>
              <div>
                <span style={{ color: '#dae2fd', fontWeight: 600 }}>Fill Quality Gate:</span>
                <p style={{ color: '#849588', marginTop: '2px', fontSize: '0.7rem' }}>
                  Certified PASS between 40.0% &ndash; 85.0% nominal limits.
                </p>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '0.25rem' }}>
            <button 
              type="button" 
              className="btn btn-secondary" 
              onClick={onClose} 
              disabled={loading}
              style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}
            >
              Cancel
            </button>
            <button 
              type="submit" 
              className="btn btn-primary" 
              disabled={loading || !file}
              style={{ 
                padding: '0.5rem 1.25rem', 
                fontSize: '0.85rem',
                background: (!loading && file) ? 'linear-gradient(to right, #0284c7, #00daf3)' : undefined,
                color: (!loading && file) ? '#050811' : undefined,
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                boxShadow: (!loading && file) ? '0 0 12px rgba(0, 218, 243, 0.35)' : 'none'
              }}
            >
              {loading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  <span>Analyzing Liquid Level...</span>
                </>
              ) : (
                <>
                  <UploadCloud size={16} />
                  <span>Run Fill Inspection</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
