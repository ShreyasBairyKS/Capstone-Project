import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

export const StationContext = createContext(null);

// POSTs a model load/unload endpoint and throws with the backend's error detail on non-2xx.
async function postStationCommand(path) {
  const res = await fetch(path, { method: 'POST' });
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json()).detail; } catch (_) { /* non-JSON error body */ }
    throw new Error(detail || `${path} failed with status ${res.status}`);
  }
}

export function StationProvider({ children }) {
  // Station 1 ('cap') is ALWAYS the primary default station on startup
  const [activeStation, setActiveStation] = useState(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const st = params.get('station');
      if (st === 'fill') return 'fill';
    }
    return 'cap';
  });

  const [isSwapping, setIsSwapping] = useState(false);
  const [swapMessage, setSwapMessage] = useState('');
  const [swapError, setSwapError] = useState('');
  const [capHealth, setCapHealth] = useState(null);
  const [fillHealth, setFillHealth] = useState(null);

  // Poll health on mount
  const refreshHealth = useCallback(async () => {
    try {
      const [cH, fH] = await Promise.all([
        fetch('/api/health').then(r => r.json()).catch(() => null),
        fetch('/api/fill/health').then(r => r.json()).catch(() => null)
      ]);
      if (cH) setCapHealth(cH);
      if (fH) setFillHealth(fH);
    } catch (e) {
      console.warn('Failed to fetch station health:', e);
    }
  }, []);

  // On first mount, make the GPU state match the station shown (e.g. a reload on ?station=fill must
  // not leave the Cap model resident next to the Fill models: Station 1 is eager-loaded at backend boot).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        if (activeStation === 'fill') {
          setIsSwapping(true);
          setSwapMessage('Syncing GPU: loading Fill Level models, releasing Cap model...');
          await postStationCommand('/api/unload');
          await postStationCommand('/api/fill/load');
        } else {
          // Idempotent: both are no-ops when the GPU state already matches
          await postStationCommand('/api/fill/unload');
          await postStationCommand('/api/load');
        }
      } catch (err) {
        console.warn('Initial station sync failed:', err);
        if (!cancelled) setSwapError(String(err.message || err));
      } finally {
        if (!cancelled) {
          setIsSwapping(false);
          setSwapMessage('');
        }
        refreshHealth();
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Symmetrical VRAM Swapping
  const switchStation = useCallback(async (targetStation) => {
    if (targetStation === activeStation || isSwapping) return;

    setIsSwapping(true);
    setSwapError('');
    // Unload the station being left, load the target; if the target fails, reload the previous
    // station so the operator is never left with no model resident.
    const unloadPath = targetStation === 'fill' ? '/api/unload' : '/api/fill/unload';
    const loadPath = targetStation === 'fill' ? '/api/fill/load' : '/api/load';
    const rollbackPath = targetStation === 'fill' ? '/api/load' : '/api/fill/load';
    const targetLabel = targetStation === 'fill' ? 'Fill Level models' : 'Cap Inspection model';
    const sourceLabel = targetStation === 'fill' ? 'Cap model' : 'Fill Level models';
    try {
      setSwapMessage(`Deallocating ${sourceLabel} from GPU VRAM...`);
      await postStationCommand(unloadPath);

      setSwapMessage(`Loading ${targetLabel} into GPU VRAM (first load compiles TensorRT engines and can take minutes)...`);
      try {
        await postStationCommand(loadPath);
      } catch (loadErr) {
        setSwapMessage('Load failed, restoring previous station...');
        await postStationCommand(rollbackPath).catch(() => null);
        throw loadErr;
      }

      setActiveStation(targetStation);

      // Update URL search param
      try {
        const url = new URL(window.location);
        url.searchParams.set('station', targetStation);
        window.history.replaceState({}, '', url);
      } catch (_) {}

      await refreshHealth();
    } catch (err) {
      console.error('Error during station switch:', err);
      setSwapError(`Station switch failed: ${err.message || err}`);
      await refreshHealth();
    } finally {
      setIsSwapping(false);
      setSwapMessage('');
    }
  }, [activeStation, isSwapping, refreshHealth]);

  return (
    <StationContext.Provider value={{
      activeStation,
      isSwapping,
      swapMessage,
      swapError,
      capHealth,
      fillHealth,
      switchStation,
      refreshHealth
    }}>
      {children}
    </StationContext.Provider>
  );
}

export function useStation() {
  const ctx = useContext(StationContext);
  if (!ctx) throw new Error('useStation must be used within StationProvider');
  return ctx;
}
