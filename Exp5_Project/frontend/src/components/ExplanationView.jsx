import React from 'react'

export default function ExplanationView({ explanation, className, loading }) {
  if (loading) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200 p-10 text-center text-slate-500 animate-pulse">
        Generating Integrated Gradients attribution map…
      </div>
    )
  }

  if (!explanation) {
    return (
      <div className="bg-white rounded-2xl border border-dashed border-slate-300 p-10 text-center text-slate-400">
        Select a predicted class and click "Explain Prediction" to visualize which
        pixels drove the model's decision.
      </div>
    )
  }

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5">
      <div className="flex items-baseline justify-between flex-wrap gap-2 mb-4">
        <h3 className="font-semibold text-slate-800">
          Explaining:{' '}
          <span className="text-brand-700 capitalize">{className?.replaceAll('_', ' ')}</span>
        </h3>
        <div className="text-xs text-slate-400 space-x-3">
          <span>Baseline: <b className="text-slate-600 capitalize">{explanation.baseline_type}</b></span>
          <span>Steps: <b className="text-slate-600">{explanation.n_steps}</b></span>
          <span>Convergence Δ: <b className="text-slate-600">{explanation.convergence_delta.toFixed(4)}</b></span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div>
          <img
            src={`data:image/png;base64,${explanation.original_resized}`}
            alt="original"
            className="rounded-lg w-full border border-slate-100"
          />
          <p className="text-xs text-center text-slate-500 mt-1">Original (224×224)</p>
        </div>
        <div>
          <img
            src={`data:image/png;base64,${explanation.heatmap}`}
            alt="heatmap"
            className="rounded-lg w-full border border-slate-100"
          />
          <p className="text-xs text-center text-slate-500 mt-1">IG Attribution Heatmap</p>
        </div>
        <div>
          <img
            src={`data:image/png;base64,${explanation.overlay}`}
            alt="overlay"
            className="rounded-lg w-full border border-slate-100"
          />
          <p className="text-xs text-center text-slate-500 mt-1">Overlay on Original</p>
        </div>
      </div>
    </div>
  )
}
