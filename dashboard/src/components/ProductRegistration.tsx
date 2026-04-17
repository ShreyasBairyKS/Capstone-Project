/**
 * ProductRegistration.tsx
 *
 * Supervisor/admin-only form to register a new product in the VisionFood QAI system.
 *
 * Fields (in order):
 *   SKU · Product Name · Description · Product Category · Product Sub-Type
 *   Container Contents · SKU Profile · Expected QR Code · Date Fields (repeating rows)
 *
 * Cascading logic:
 *   - Sub-type dropdown is filtered by selected category.
 *   - Container contents is auto-locked based on selected sub-type rules.
 */

import { useState, useEffect, useCallback } from 'react'
import type { ProductCategory, ProductSubType, ContainerContents, ExpectedDateField } from '../types'
import { createProduct, getSkuProfiles } from '../api'

// ---------------------------------------------------------------------------
// Cascade maps
// ---------------------------------------------------------------------------

const SUB_TYPES_BY_CATEGORY: Record<ProductCategory, ProductSubType[]> = {
  beverage: ['transparent_bottle', 'rigid_can'],
  food: ['flexible_wrapper', 'rigid_can', 'rigid_box'],
  general: ['transparent_bottle', 'rigid_can', 'flexible_wrapper', 'rigid_box'],
}

const CONTENTS_FOR_SUBTYPE: Record<ProductSubType, ContainerContents> = {
  transparent_bottle: 'liquid',
  rigid_can: 'liquid',   // overridden to 'solid' for food category — handled in submit
  flexible_wrapper: 'solid',
  rigid_box: 'solid',
}

const SUBTYPE_LABEL: Record<ProductSubType, string> = {
  transparent_bottle: 'Transparent Bottle',
  rigid_can: 'Rigid Can',
  flexible_wrapper: 'Flexible Wrapper (Pouch / Bag)',
  rigid_box: 'Rigid Box (Carton)',
}

// ---------------------------------------------------------------------------
// SKU regex
// ---------------------------------------------------------------------------
const SKU_RE = /^[a-z0-9_]{3,64}$/

// ---------------------------------------------------------------------------
// Empty date-field row
// ---------------------------------------------------------------------------
const emptyDate = (): ExpectedDateField => ({ name: '', format: '', value: '' })

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface FormState {
  sku: string
  name: string
  description: string
  product_category: ProductCategory | ''
  product_sub_type: ProductSubType | ''
  container_contents: ContainerContents | ''
  sku_profile_name: string
  qr_code: string
  expected_dates: ExpectedDateField[]
}

const INITIAL: FormState = {
  sku: '',
  name: '',
  description: '',
  product_category: '',
  product_sub_type: '',
  container_contents: '',
  sku_profile_name: '',
  qr_code: '',
  expected_dates: [],
}

export function ProductRegistration() {
  const [form, setForm] = useState<FormState>(INITIAL)
  const [skuProfiles, setSkuProfiles] = useState<string[]>([])
  const [profilesLoading, setProfilesLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null)
  const [errors, setErrors] = useState<Record<string, string>>({})

  // Load SKU profiles from API
  useEffect(() => {
    getSkuProfiles()
      .then(setSkuProfiles)
      .catch(() => setSkuProfiles([]))
      .finally(() => setProfilesLoading(false))
  }, [])

  // Dismiss toast after 4 s
  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 4000)
    return () => clearTimeout(t)
  }, [toast])

  // ---------------------------------------------------------------------------
  // Field change handlers
  // ---------------------------------------------------------------------------

  const handleField = (key: keyof FormState, value: string) => {
    setErrors((e) => ({ ...e, [key]: '' }))
    setForm((f) => ({ ...f, [key]: value }))
  }

  const handleCategory = (cat: ProductCategory | '') => {
    setForm((f) => ({ ...f, product_category: cat, product_sub_type: '', container_contents: '' }))
    setErrors((e) => ({ ...e, product_category: '', product_sub_type: '', container_contents: '' }))
  }

  const handleSubType = (sub: ProductSubType | '') => {
    const contents = sub ? CONTENTS_FOR_SUBTYPE[sub] : ''
    // For rigid_can in food category, override to solid
    const resolvedContents =
      sub === 'rigid_can' && form.product_category === 'food' ? 'solid' : contents
    setForm((f) => ({ ...f, product_sub_type: sub, container_contents: resolvedContents }))
    setErrors((e) => ({ ...e, product_sub_type: '', container_contents: '' }))
  }

  // ---------------------------------------------------------------------------
  // Date field rows
  // ---------------------------------------------------------------------------

  const addDateField = () =>
    setForm((f) => ({ ...f, expected_dates: [...f.expected_dates, emptyDate()] }))

  const removeDateField = (i: number) =>
    setForm((f) => ({ ...f, expected_dates: f.expected_dates.filter((_, idx) => idx !== i) }))

  const updateDateField = (i: number, key: keyof ExpectedDateField, val: string) =>
    setForm((f) => {
      const rows = [...f.expected_dates]
      rows[i] = { ...rows[i], [key]: val }
      return { ...f, expected_dates: rows }
    })

  // ---------------------------------------------------------------------------
  // Validation
  // ---------------------------------------------------------------------------

  const validate = (): boolean => {
    const errs: Record<string, string> = {}
    if (!SKU_RE.test(form.sku))
      errs.sku = 'SKU must be 3–64 lowercase letters, digits, or underscores.'
    if (!form.name.trim()) errs.name = 'Product name is required.'
    if (!form.product_category) errs.product_category = 'Required.'
    if (!form.product_sub_type) errs.product_sub_type = 'Required.'
    if (!form.container_contents) errs.container_contents = 'Required.'
    if (!form.sku_profile_name) errs.sku_profile_name = 'Required.'

    form.expected_dates.forEach((d, i) => {
      if (!d.name.trim()) errs[`date_name_${i}`] = 'Field name required.'
      if (!d.format.trim()) errs[`date_format_${i}`] = 'Format required.'
    })

    setErrors(errs)
    return Object.keys(errs).length === 0
  }

  // ---------------------------------------------------------------------------
  // Submit
  // ---------------------------------------------------------------------------

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!validate()) return
    setSubmitting(true)
    try {
      await createProduct({
        sku: form.sku,
        name: form.name.trim(),
        description: form.description.trim() || undefined,
        product_category: form.product_category as ProductCategory,
        product_sub_type: form.product_sub_type as ProductSubType,
        container_contents: form.container_contents as ContainerContents,
        sku_profile_name: form.sku_profile_name,
        qr_code: form.qr_code.trim() || undefined,
        expected_dates: form.expected_dates.map((d) => ({
          name: d.name.trim(),
          format: d.format.trim(),
          value: d.value?.trim() || undefined,
        })),
      })
      setToast({ type: 'success', message: `Product "${form.name}" registered successfully.` })
      setForm(INITIAL)
      setErrors({})
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Registration failed.'
      const is409 = msg.includes('409')
      if (is409) {
        setErrors((e) => ({ ...e, sku: `SKU '${form.sku}' is already registered.` }))
      }
      setToast({ type: 'error', message: is409 ? `SKU '${form.sku}' already exists.` : msg })
    } finally {
      setSubmitting(false)
    }
  }

  // ---------------------------------------------------------------------------
  // Render helpers
  // ---------------------------------------------------------------------------

  const field = (key: string) =>
    errors[key] ? (
      <p className="text-red-400 text-xs mt-1">{errors[key]}</p>
    ) : null

  const availableSubTypes =
    form.product_category ? SUB_TYPES_BY_CATEGORY[form.product_category] : []

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="bg-gray-900 rounded-xl p-6 space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold text-gray-100">Register New Product</h3>
          <p className="text-xs text-gray-500 mt-0.5">
            Supervisor / Admin only · Creates a MongoDB product document
          </p>
        </div>
        {toast && (
          <div
            className={`text-xs px-3 py-2 rounded-lg font-medium transition-all ${
              toast.type === 'success'
                ? 'bg-emerald-900/60 text-emerald-300 border border-emerald-700'
                : 'bg-red-900/60 text-red-300 border border-red-700'
            }`}
          >
            {toast.message}
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4" id="product-registration-form">

        {/* ── Row 1: SKU + Product Name ── */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label htmlFor="reg-sku" className="block text-xs font-medium text-gray-400 mb-1">
              SKU <span className="text-red-400">*</span>
            </label>
            <input
              id="reg-sku"
              type="text"
              placeholder="e.g. bottle_250ml"
              value={form.sku}
              onChange={(e) => handleField('sku', e.target.value.toLowerCase())}
              className={`w-full bg-gray-800 text-gray-100 text-sm rounded-lg px-3 py-2.5 border focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                errors.sku ? 'border-red-500' : 'border-gray-700'
              }`}
            />
            {field('sku')}
          </div>

          <div>
            <label htmlFor="reg-name" className="block text-xs font-medium text-gray-400 mb-1">
              Product Name <span className="text-red-400">*</span>
            </label>
            <input
              id="reg-name"
              type="text"
              placeholder="e.g. Standard 250ml Glass Bottle"
              value={form.name}
              onChange={(e) => handleField('name', e.target.value)}
              className={`w-full bg-gray-800 text-gray-100 text-sm rounded-lg px-3 py-2.5 border focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                errors.name ? 'border-red-500' : 'border-gray-700'
              }`}
            />
            {field('name')}
          </div>
        </div>

        {/* ── Description ── */}
        <div>
          <label htmlFor="reg-description" className="block text-xs font-medium text-gray-400 mb-1">
            Description
          </label>
          <textarea
            id="reg-description"
            rows={2}
            placeholder="Optional product description"
            value={form.description}
            onChange={(e) => handleField('description', e.target.value)}
            className="w-full bg-gray-800 text-gray-100 text-sm rounded-lg px-3 py-2.5 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          />
        </div>

        {/* ── Row 2: Category → Sub-Type → Contents ── */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label htmlFor="reg-category" className="block text-xs font-medium text-gray-400 mb-1">
              Product Category <span className="text-red-400">*</span>
            </label>
            <select
              id="reg-category"
              value={form.product_category}
              onChange={(e) => handleCategory(e.target.value as ProductCategory | '')}
              className={`w-full bg-gray-800 text-gray-100 text-sm rounded-lg px-3 py-2.5 border focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                errors.product_category ? 'border-red-500' : 'border-gray-700'
              }`}
            >
              <option value="">Select…</option>
              <option value="beverage">Beverage</option>
              <option value="food">Food</option>
              <option value="general">General</option>
            </select>
            {field('product_category')}
          </div>

          <div>
            <label htmlFor="reg-subtype" className="block text-xs font-medium text-gray-400 mb-1">
              Product Sub-Type <span className="text-red-400">*</span>
            </label>
            <select
              id="reg-subtype"
              value={form.product_sub_type}
              onChange={(e) => handleSubType(e.target.value as ProductSubType | '')}
              disabled={!form.product_category}
              className={`w-full bg-gray-800 text-gray-100 text-sm rounded-lg px-3 py-2.5 border focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 ${
                errors.product_sub_type ? 'border-red-500' : 'border-gray-700'
              }`}
            >
              <option value="">Select…</option>
              {availableSubTypes.map((st) => (
                <option key={st} value={st}>{SUBTYPE_LABEL[st]}</option>
              ))}
            </select>
            {field('product_sub_type')}
          </div>

          <div>
            <label htmlFor="reg-contents" className="block text-xs font-medium text-gray-400 mb-1">
              Container Contents <span className="text-red-400">*</span>
            </label>
            <input
              id="reg-contents"
              type="text"
              value={form.container_contents}
              readOnly
              placeholder="Auto-set from sub-type"
              className="w-full bg-gray-700/60 text-gray-300 text-sm rounded-lg px-3 py-2.5 border border-gray-700 cursor-not-allowed"
            />
            {field('container_contents')}
          </div>
        </div>

        {/* ── Row 3: SKU Profile + QR Code ── */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label htmlFor="reg-profile" className="block text-xs font-medium text-gray-400 mb-1">
              SKU Profile <span className="text-red-400">*</span>
            </label>
            <select
              id="reg-profile"
              value={form.sku_profile_name}
              onChange={(e) => handleField('sku_profile_name', e.target.value)}
              disabled={profilesLoading}
              className={`w-full bg-gray-800 text-gray-100 text-sm rounded-lg px-3 py-2.5 border focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 ${
                errors.sku_profile_name ? 'border-red-500' : 'border-gray-700'
              }`}
            >
              <option value="">{profilesLoading ? 'Loading profiles…' : 'Select profile…'}</option>
              {skuProfiles.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
            {field('sku_profile_name')}
          </div>

          <div>
            <label htmlFor="reg-qr" className="block text-xs font-medium text-gray-400 mb-1">
              Expected QR Code
            </label>
            <input
              id="reg-qr"
              type="text"
              placeholder="QR code value for barcode verification"
              value={form.qr_code}
              onChange={(e) => handleField('qr_code', e.target.value)}
              className="w-full bg-gray-800 text-gray-100 text-sm rounded-lg px-3 py-2.5 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* ── OCR Date Fields ── */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-medium text-gray-400">OCR Date Fields</label>
            <button
              type="button"
              onClick={addDateField}
              className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
              id="add-date-field-btn"
            >
              + Add field
            </button>
          </div>

          {form.expected_dates.length === 0 && (
            <p className="text-xs text-gray-600 italic">No date fields — click "Add field" to add one.</p>
          )}

          <div className="space-y-2">
            {form.expected_dates.map((d, i) => (
              <div key={i} className="grid grid-cols-10 gap-2 items-start">
                <div className="col-span-3">
                  <input
                    id={`date-name-${i}`}
                    type="text"
                    placeholder="Field name (e.g. expiry_date)"
                    value={d.name}
                    onChange={(e) => updateDateField(i, 'name', e.target.value)}
                    className={`w-full bg-gray-800 text-gray-100 text-xs rounded-lg px-2.5 py-2 border focus:outline-none focus:ring-1 focus:ring-blue-500 ${
                      errors[`date_name_${i}`] ? 'border-red-500' : 'border-gray-700'
                    }`}
                  />
                  {errors[`date_name_${i}`] && (
                    <p className="text-red-400 text-xs mt-0.5">{errors[`date_name_${i}`]}</p>
                  )}
                </div>
                <div className="col-span-3">
                  <input
                    id={`date-format-${i}`}
                    type="text"
                    placeholder="Format (e.g. MM/YYYY)"
                    value={d.format}
                    onChange={(e) => updateDateField(i, 'format', e.target.value)}
                    className={`w-full bg-gray-800 text-gray-100 text-xs rounded-lg px-2.5 py-2 border focus:outline-none focus:ring-1 focus:ring-blue-500 ${
                      errors[`date_format_${i}`] ? 'border-red-500' : 'border-gray-700'
                    }`}
                  />
                  {errors[`date_format_${i}`] && (
                    <p className="text-red-400 text-xs mt-0.5">{errors[`date_format_${i}`]}</p>
                  )}
                </div>
                <div className="col-span-3">
                  <input
                    id={`date-value-${i}`}
                    type="text"
                    placeholder="Expected value (optional)"
                    value={d.value ?? ''}
                    onChange={(e) => updateDateField(i, 'value', e.target.value)}
                    className="w-full bg-gray-800 text-gray-100 text-xs rounded-lg px-2.5 py-2 border border-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                </div>
                <div className="col-span-1 flex items-center justify-center pt-1">
                  <button
                    type="button"
                    onClick={() => removeDateField(i)}
                    id={`remove-date-${i}`}
                    className="text-gray-600 hover:text-red-400 text-xs transition-colors"
                    title="Remove"
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Submit ── */}
        <div className="flex justify-end pt-2">
          <button
            type="submit"
            id="product-registration-submit"
            disabled={submitting}
            className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white text-sm font-semibold rounded-lg transition-colors min-h-[40px] min-w-[140px]"
          >
            {submitting ? 'Registering…' : 'Register Product'}
          </button>
        </div>
      </form>
    </div>
  )
}
