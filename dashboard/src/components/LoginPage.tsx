import { useState, memo } from 'react'
import { Shield, Eye, EyeOff, AlertTriangle, Cpu } from 'lucide-react'
import { useApp } from '../store'
import type { UserRole, AuthUser } from '../store'

const DEMO_USERS: Record<string, { password: string; role: UserRole }> = {
  operator:   { password: 'op1234',    role: 'operator' },
  supervisor: { password: 'sup1234',   role: 'supervisor' },
  admin:      { password: 'admin1234', role: 'admin' },
}

export const LoginPage = memo(function LoginPage() {
  const { dispatch } = useApp()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    await new Promise((r) => setTimeout(r, 400))
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
    <div className="min-h-screen flex bg-gray-950">
      {/* Left panel - branding */}
      <div className="hidden lg:flex lg:w-[480px] xl:w-[540px] flex-col justify-between p-12 bg-gradient-to-br from-gray-900 via-gray-950 to-gray-900 border-r border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-brand-500 rounded-xl flex items-center justify-center font-extrabold text-gray-950 text-sm shadow-glow-green">
            VF
          </div>
          <span className="text-white font-bold text-sm tracking-tight">VisionFood QAI</span>
        </div>

        <div className="space-y-6">
          <div className="inline-flex items-center gap-2 text-brand-400 text-xs font-semibold uppercase tracking-widest bg-brand-500/10 border border-brand-500/20 px-3 py-1.5 rounded-full">
            <Cpu size={12} /> AI-Powered Quality Inspection
          </div>
          <h2 className="text-4xl font-extrabold text-white leading-tight tracking-tight">
            Industrial<br />Quality<br />Intelligence
          </h2>
          <p className="text-gray-400 text-sm leading-relaxed max-w-xs">
            Real-time defect detection, severity triage, and quality analytics powered by computer vision for food and beverage production lines.
          </p>

          <div className="grid grid-cols-2 gap-3 pt-2">
            {[
              { label: 'Cap Detection', desc: 'YOLO v11 2-pass' },
              { label: 'Fill Level', desc: 'EfficientViT-based' },
              { label: 'Tear Detection', desc: 'Wrapper integrity' },
              { label: 'Label QR', desc: 'Barcode validation' },
            ].map(({ label, desc }) => (
              <div key={label} className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-3">
                <p className="text-white text-xs font-semibold">{label}</p>
                <p className="text-gray-500 text-[11px] mt-0.5">{desc}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="dot-online" />
          <span className="text-gray-500 text-xs">System operational</span>
        </div>
      </div>

      {/* Right panel - login form */}
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-sm">
          {/* Mobile logo */}
          <div className="flex flex-col items-center mb-8 lg:hidden">
            <div className="w-12 h-12 bg-brand-500 rounded-2xl flex items-center justify-center font-extrabold text-gray-950 text-xl mb-3 shadow-glow-green">
              VF
            </div>
            <h1 className="text-xl font-bold text-white">VisionFood QAI</h1>
            <p className="text-gray-500 text-sm mt-1">Quality Intelligence Dashboard</p>
          </div>

          <div className="mb-8">
            <h2 className="text-2xl font-bold text-white">Sign in</h2>
            <p className="text-gray-500 text-sm mt-1">Access the quality inspection dashboard</p>
          </div>

          <form onSubmit={handleLogin} className="space-y-4" noValidate>
            <div className="space-y-1">
              <label htmlFor="login-user" className="block text-xs text-gray-400 font-medium">
                Username
              </label>
              <input
                id="login-user"
                type="text"
                autoComplete="username"
                value={username}
                onChange={(e) => { setUsername(e.target.value); setError('') }}
                required
                className="input w-full"
                placeholder="operator / supervisor / admin"
                autoFocus
              />
            </div>

            <div className="space-y-1">
              <label htmlFor="login-pass" className="block text-xs text-gray-400 font-medium">
                Password
              </label>
              <div className="relative">
                <input
                  id="login-pass"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => { setPassword(e.target.value); setError('') }}
                  required
                  className="input w-full pr-10"
                  placeholder="Enter password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 transition-colors"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </div>

            {error && (
              <div role="alert" className="flex items-center gap-2 text-red-400 text-xs bg-red-500/10 border border-red-500/20 rounded-xl px-3 py-2.5">
                <AlertTriangle size={12} className="flex-shrink-0" />
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !username || !password}
              className="btn btn-primary w-full mt-2"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Authenticating...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  <Shield size={14} />
                  Sign In
                </span>
              )}
            </button>
          </form>

          {/* Demo hint */}
          <div className="mt-6 p-4 bg-gray-900 border border-gray-800 rounded-xl">
            <p className="text-xs text-gray-500 font-medium mb-2">Demo credentials</p>
            <div className="space-y-1">
              {Object.entries(DEMO_USERS).map(([u, { password: p, role }]) => (
                <button
                  key={u}
                  type="button"
                  onClick={() => { setUsername(u); setPassword(p); setError('') }}
                  className="w-full flex items-center justify-between px-3 py-1.5 rounded-lg hover:bg-gray-800 transition-colors group text-left"
                >
                  <span className="text-gray-300 text-xs font-mono">{u}</span>
                  <span className="text-gray-600 text-[11px] capitalize group-hover:text-gray-400 transition-colors">{role}</span>
                </button>
              ))}
            </div>
          </div>

          <p className="text-center text-gray-700 text-xs mt-6">
            VisionFood QAI &middot; Capstone 2026
          </p>
        </div>
      </div>
    </div>
  )
})
