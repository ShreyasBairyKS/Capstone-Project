import { useState, memo } from 'react'
import { useApp } from '../store'
import type { UserRole, AuthUser } from '../store'

// Simulated credential check (replace with real API call when backend auth is ready)
const DEMO_USERS: Record<string, { password: string; role: UserRole }> = {
  operator:   { password: 'op1234',   role: 'operator' },
  supervisor: { password: 'sup1234',  role: 'supervisor' },
  admin:      { password: 'admin1234',role: 'admin' },
}

export const LoginPage = memo(function LoginPage() {
  const { dispatch } = useApp()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    // Simulate async auth
    await new Promise((r) => setTimeout(r, 300))
    const user = DEMO_USERS[username.toLowerCase()]
    if (user && user.password === password) {
      const auth: AuthUser = { username, role: user.role, token: `demo-token-${Date.now()}` }
      dispatch({ type: 'SET_AUTH', payload: auth })
    } else {
      setError('Invalid username or password')
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950 p-4">
      <div className="w-full max-w-sm bg-gray-900 rounded-2xl p-8 border border-gray-800 shadow-2xl">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-12 h-12 bg-green-500 rounded-2xl flex items-center justify-center font-extrabold text-black text-xl mb-3">
            VF
          </div>
          <h1 className="text-xl font-bold text-white">VisionFood QAI</h1>
          <p className="text-gray-500 text-sm mt-1">Quality Intelligence Dashboard</p>
        </div>

        <form onSubmit={handleLogin} className="space-y-4" noValidate>
          <div>
            <label htmlFor="login-user" className="block text-xs text-gray-400 mb-1">Username</label>
            <input
              id="login-user"
              type="text"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              className="w-full bg-gray-800 text-gray-100 rounded-lg px-3 py-2.5 text-sm border border-gray-700 focus:outline-none focus:border-blue-500 transition-colors"
              placeholder="operator / supervisor / admin"
            />
          </div>
          <div>
            <label htmlFor="login-pass" className="block text-xs text-gray-400 mb-1">Password</label>
            <input
              id="login-pass"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full bg-gray-800 text-gray-100 rounded-lg px-3 py-2.5 text-sm border border-gray-700 focus:outline-none focus:border-blue-500 transition-colors"
              placeholder="••••••••"
            />
          </div>

          {error && (
            <p role="alert" className="text-red-400 text-xs bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-700 text-white font-semibold rounded-lg py-2.5 text-sm transition-colors min-h-[44px] focus:outline-none focus-visible:ring-2 focus-visible:ring-green-500"
          >
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        <p className="text-gray-600 text-xs text-center mt-6">
          Demo credentials — operator/op1234, supervisor/sup1234, admin/admin1234
        </p>
      </div>
    </div>
  )
})
