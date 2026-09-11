import React from 'react'

export default function PredictionsPanel({ predictions, selectedIdx, onSelect, loading }) {
  if (loading) {
    return <div className="text-slate-500 animate-pulse text-sm">Running ResNet18 inference…</div>
  }
  if (!predictions || predictions.length === 0) {
    return (
      <div className="text-slate-400 text-sm">
        Upload an image and click "Predict" to see the top-5 ImageNet-1K classes.
      </div>
    )
  }

  const maxConf = predictions[0].confidence

  return (
    <div className="space-y-3">
      {predictions.map((p) => {
        const isSelected = p.class_idx === selectedIdx
        return (
          <button
            key={p.class_idx}
            onClick={() => onSelect(p.class_idx)}
            className={`w-full text-left p-3 rounded-xl border transition-all
              ${isSelected ? 'border-brand-500 bg-brand-50 ring-2 ring-brand-200' : 'border-slate-200 hover:border-brand-300 bg-white'}`}
          >
            <div className="flex justify-between items-center mb-1">
              <span className="font-medium capitalize text-slate-800">
                {p.class_name.replaceAll('_', ' ')}
              </span>
              <span className="text-sm font-semibold text-brand-700">
                {(p.confidence * 100).toFixed(2)}%
              </span>
            </div>
            <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-brand-500 to-brand-700"
                style={{ width: `${(p.confidence / maxConf) * 100}%` }}
              />
            </div>
          </button>
        )
      })}
    </div>
  )
}
