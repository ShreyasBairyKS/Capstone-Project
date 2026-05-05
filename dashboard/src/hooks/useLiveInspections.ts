import { useEffect, useRef, useCallback } from 'react'
import { useAppDispatch } from '../store'
import type { InspectionResult } from '../types'

const WS_URL = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000/ws/live'
const RECONNECT_MS = 3000

/**
 * Singleton hook: connects to the backend WS, dispatches SET_LIVE / SET_WS_CONNECTED.
 * Mount once at the App level.
 */
export function useLiveInspections() {
  const dispatch = useAppDispatch()
  const ws = useRef<WebSocket | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const connect = useCallback(() => {
    if (ws.current) ws.current.close()

    const sock = new WebSocket(WS_URL)
    ws.current = sock

    sock.onopen = () => dispatch({ type: 'SET_WS_CONNECTED', payload: true })

    sock.onmessage = (evt) => {
      try {
        const payload = JSON.parse(evt.data) as InspectionResult
        dispatch({ type: 'SET_LIVE', payload })
      } catch {
        // ignore malformed
      }
    }

    sock.onclose = () => {
      dispatch({ type: 'SET_WS_CONNECTED', payload: false })
      timer.current = setTimeout(connect, RECONNECT_MS)
    }

    sock.onerror = () => sock.close()
  }, [dispatch])

  useEffect(() => {
    connect()
    return () => {
      if (timer.current) clearTimeout(timer.current)
      ws.current?.close()
    }
  }, [connect])
}
