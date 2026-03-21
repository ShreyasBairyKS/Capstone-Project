import { useEffect, useRef, useState, useCallback } from 'react'
import type { InspectionResult } from './types'

/**
 * Hook that connects to the /ws/live WebSocket endpoint and streams
 * live inspection events. Returns the latest event and connection status.
 */
export function useLiveStream(wsUrl: string) {
  const [latest, setLatest] = useState<InspectionResult | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const connect = useCallback(() => {
    if (wsRef.current) wsRef.current.close()

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      setError(null)
    }

    ws.onmessage = (evt) => {
      try {
        const payload = JSON.parse(evt.data) as InspectionResult
        setLatest(payload)
      } catch {
        // ignore malformed messages
      }
    }

    ws.onclose = () => {
      setConnected(false)
      // Auto-reconnect after 3 s
      reconnectTimer.current = setTimeout(connect, 3000)
    }

    ws.onerror = () => {
      setError('WebSocket connection failed')
      ws.close()
    }
  }, [wsUrl])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { latest, connected, error }
}
