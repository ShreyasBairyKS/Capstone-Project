import { useEffect, useState, useCallback } from 'react'
import { Package, Plus, X, Search, Tag, AlertTriangle } from 'lucide-react'
import { getProducts, createProduct } from '../api'
import { useToast } from '../store'
import type { Product, ProductCreate, ProductCategory, ProductSubType, ContainerContents } from '../types'

const CATEGORY_COLORS: Record<ProductCategory | string, string> = {
  beverage: 'text-blue-400 bg-blue-500/10 border-blue-500/30',
  food: 'text-orange-400 bg-orange-500/10 border-orange-500/30',
  general: 'text-gray-400 bg-gray-500/10 border-gray-500/30',
}

function ProductCard({ product }: { product: Product }) {
  const catColor = CATEGORY_COLORS[product.product_category] ?? CATEGORY_COLORS.general
  return (
    <div className="card-sm hover:border-gray-600 transition-colors group">
      <div className="flex items-start gap-3 mb-3">
        <div className="w-10 h-10 bg-gray-800 rounded-xl flex items-center justify-center flex-shrink-0 border border-gray-700 group-hover:border-brand-500/40 transition-colors">
          <Package size={16} className="text-gray-400 group-hover:text-brand-400 transition-colors" />
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-gray-200 truncate">{product.name}</h3>
          <p className="text-xs text-gray-500 font-mono truncate">{product.sku}</p>
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5 mt-2">
        <span className={`inline-flex items-center gap-1 text-[11px] font-medium px-2 py-0.5 rounded-full border ${catColor}`}>
          <Tag size={9} />{product.product_category}
        </span>
        <span className="inline-flex items-center text-[11px] font-medium px-2 py-0.5 rounded-full border text-gray-400 bg-gray-700/30 border-gray-700">
          {product.product_sub_type.replace(/_/g, ' ')}
        </span>
        <span className="inline-flex items-center text-[11px] font-medium px-2 py-0.5 rounded-full border text-gray-500 bg-gray-700/20 border-gray-700">
          {product.container_contents}
        </span>
      </div>
      {product.description && (
        <p className="text-xs text-gray-500 mt-2 line-clamp-2">{product.description}</p>
      )}
    </div>
  )
}

function RegisterModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const toast = useToast()
  const [form, setForm] = useState<ProductCreate>({
    name: '', sku: '',
    product_category: 'beverage',
    product_sub_type: 'transparent_bottle',
    container_contents: 'liquid',
    sku_profile_name: '',
    description: '',
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function set<K extends keyof ProductCreate>(key: K, value: ProductCreate[K]) {
    setForm((f) => ({ ...f, [key]: value }))
    setError(null)
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (!form.name.trim() || !form.sku.trim()) { setError('Name and SKU are required'); return }
    if (!form.sku_profile_name.trim()) { setError('SKU profile name is required'); return }
    setLoading(true)
    try {
      await createProduct(form)
      toast('success', 'Product registered', `${form.name} has been added to the catalog`)
      onCreated()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  function handleBackdrop(e: React.MouseEvent<HTMLDivElement>) {
    if (e.target === e.currentTarget) onClose()
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4"
      role="dialog"
      aria-modal
      onClick={handleBackdrop}
    >
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-md shadow-2xl animate-bounce-in">
        <div className="flex items-center gap-3 px-5 py-4 border-b border-gray-800">
          <Package size={15} className="text-brand-400" />
          <h2 className="text-sm font-semibold text-white flex-1">Register Product</h2>
          <button onClick={onClose} className="btn-icon text-gray-400 hover:text-white border-gray-700" aria-label="Close">
            <X size={14} />
          </button>
        </div>

        <form onSubmit={submit} className="p-5 space-y-4">
          <div className="space-y-1">
            <label className="text-xs text-gray-400 block" htmlFor="reg-name">Product Name *</label>
            <input id="reg-name" type="text" value={form.name}
              onChange={(e) => set('name', e.target.value)}
              className="input w-full" placeholder="e.g. Sparkling Water 500ml" autoFocus />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs text-gray-400 block" htmlFor="reg-sku">SKU *</label>
              <input id="reg-sku" type="text" value={form.sku}
                onChange={(e) => set('sku', e.target.value)}
                className="input w-full font-mono" placeholder="SKU-001" />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-gray-400 block" htmlFor="reg-profile">SKU Profile *</label>
              <input id="reg-profile" type="text" value={form.sku_profile_name}
                onChange={(e) => set('sku_profile_name', e.target.value)}
                className="input w-full" placeholder="e.g. bottle_250ml" />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs text-gray-400 block" htmlFor="reg-cat">Category</label>
              <select id="reg-cat" value={form.product_category}
                onChange={(e) => set('product_category', e.target.value as ProductCategory)}
                className="select w-full">
                {(['beverage', 'food', 'general'] as ProductCategory[]).map((c) => (
                  <option key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs text-gray-400 block" htmlFor="reg-sub">Sub-type</label>
              <select id="reg-sub" value={form.product_sub_type}
                onChange={(e) => set('product_sub_type', e.target.value as ProductSubType)}
                className="select w-full">
                {(['transparent_bottle', 'rigid_can', 'flexible_wrapper', 'rigid_box'] as ProductSubType[]).map((s) => (
                  <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="space-y-1">
            <label className="text-xs text-gray-400 block" htmlFor="reg-contents">Container Contents</label>
            <select id="reg-contents" value={form.container_contents}
              onChange={(e) => set('container_contents', e.target.value as ContainerContents)}
              className="select w-full">
              {(['liquid', 'solid'] as ContainerContents[]).map((c) => (
                <option key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-xs text-gray-400 block" htmlFor="reg-desc">Description</label>
            <input id="reg-desc" type="text" value={form.description ?? ''}
              onChange={(e) => set('description', e.target.value)}
              className="input w-full" placeholder="Optional description" />
          </div>

          {error && (
            <p className="text-red-400 text-xs flex items-center gap-1.5">
              <AlertTriangle size={11} /> {error}
            </p>
          )}

          <div className="flex gap-3 pt-2">
            <button type="button" onClick={onClose} className="btn btn-secondary flex-1">Cancel</button>
            <button type="submit" disabled={loading} className="btn btn-primary flex-1">
              {loading ? (
                <span className="flex items-center gap-2 justify-center">
                  <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Saving...
                </span>
              ) : 'Register Product'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export function ProductsPage() {
  const [products, setProducts] = useState<Product[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [showRegister, setShowRegister] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getProducts()
      setProducts(data)
    } catch {
      setProducts([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const filtered = products.filter((p) => {
    const q = search.toLowerCase()
    return !q || p.name.toLowerCase().includes(q) || p.sku.toLowerCase().includes(q) || p.product_category.toLowerCase().includes(q)
  })

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="page-title">Product Catalog</h1>
          <p className="page-sub">Manage registered products and SKU profiles</p>
        </div>
        <button onClick={() => setShowRegister(true)} className="btn btn-primary">
          <Plus size={14} />
          Register Product
        </button>
      </div>

      <div className="relative max-w-xs">
        <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" aria-hidden />
        <input type="text" value={search} onChange={(e) => setSearch(e.target.value)}
          placeholder="Search products..." className="input pl-9 w-full" />
      </div>

      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="card-sm animate-pulse space-y-3">
              <div className="flex gap-3">
                <div className="w-10 h-10 bg-gray-800 rounded-xl" />
                <div className="flex-1 space-y-1.5">
                  <div className="h-3.5 bg-gray-800 rounded w-3/4" />
                  <div className="h-3 bg-gray-800 rounded w-1/2" />
                </div>
              </div>
              <div className="h-5 bg-gray-800 rounded w-1/3" />
            </div>
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-gray-600 gap-3">
          <Package size={40} />
          <p className="text-sm font-medium">
            {search ? 'No products match your search' : 'No products registered yet'}
          </p>
          {!search && (
            <button onClick={() => setShowRegister(true)} className="btn btn-secondary text-xs mt-1">
              <Plus size={12} /> Register first product
            </button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filtered.map((p) => (
            <ProductCard key={p._id ?? p.sku} product={p} />
          ))}
        </div>
      )}

      {showRegister && (
        <RegisterModal onClose={() => setShowRegister(false)} onCreated={load} />
      )}
    </div>
  )
}
