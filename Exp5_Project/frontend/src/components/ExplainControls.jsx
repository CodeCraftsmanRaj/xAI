import React from 'react'

const BASELINES = [
  { value: 'black', label: 'Black image (zeros)' },
  { value: 'white', label: 'White image' },
  { value: 'noise', label: 'Random noise' },
  { value: 'blur', label: 'Blurred version of input' },
]

export default function ExplainControls({
  baseline, setBaseline, steps, setSteps, onExplain, disabled, loading,
}) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 space-y-4">
      <h2 className="font-semibold text-slate-800">Integrated Gradients settings</h2>

      <div>
        <label className="text-sm font-medium text-slate-700">Baseline (reference input)</label>
        <select
          value={baseline}
          onChange={(e) => setBaseline(e.target.value)}
          className="mt-1 w-full border border-slate-300 rounded-lg p-2 text-sm"
        >
          {BASELINES.map((b) => (
            <option key={b.value} value={b.value}>
              {b.label}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="text-sm font-medium text-slate-700">
          Integration steps: <span className="text-brand-700 font-semibold">{steps}</span>
        </label>
        <input
          type="range"
          min={10}
          max={200}
          step={10}
          value={steps}
          onChange={(e) => setSteps(Number(e.target.value))}
          className="w-full accent-brand-600"
        />
        <p className="text-xs text-slate-400 mt-1">More steps = smoother, slower attribution.</p>
      </div>

      <button
        onClick={onExplain}
        disabled={disabled || loading}
        className="w-full py-2.5 rounded-lg bg-brand-600 hover:bg-brand-700 disabled:bg-slate-300 text-white font-medium transition-colors"
      >
        {loading ? 'Computing Integrated Gradients…' : 'Explain Prediction'}
      </button>
    </div>
  )
}
