import React, { useState } from 'react'
import UploadCard from './components/UploadCard.jsx'
import PredictionsPanel from './components/PredictionsPanel.jsx'
import ExplainControls from './components/ExplainControls.jsx'
import ExplanationView from './components/ExplanationView.jsx'

const API_BASE = 'http://localhost:8000'

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)

  const [predictions, setPredictions] = useState([])
  const [selectedIdx, setSelectedIdx] = useState(null)
  const [predictLoading, setPredictLoading] = useState(false)

  const [baseline, setBaseline] = useState('black')
  const [steps, setSteps] = useState(50)
  const [explanation, setExplanation] = useState(null)
  const [explainLoading, setExplainLoading] = useState(false)

  const [error, setError] = useState(null)

  const handleImageSelected = (f) => {
    setFile(f)
    setPreviewUrl(URL.createObjectURL(f))
    setPredictions([])
    setSelectedIdx(null)
    setExplanation(null)
    setError(null)
  }

  const handlePredict = async () => {
    if (!file) return
    setPredictLoading(true)
    setError(null)
    setExplanation(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: formData })
      if (!res.ok) throw new Error('Prediction request failed. Is the backend running on port 8000?')
      const data = await res.json()
      setPredictions(data.predictions)
      setSelectedIdx(data.predictions[0]?.class_idx ?? null)
    } catch (e) {
      setError(e.message)
    } finally {
      setPredictLoading(false)
    }
  }

  const handleExplain = async () => {
    if (!file || selectedIdx === null) return
    setExplainLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('target_idx', selectedIdx)
      formData.append('baseline_type', baseline)
      formData.append('n_steps', steps)
      const res = await fetch(`${API_BASE}/explain`, { method: 'POST', body: formData })
      if (!res.ok) throw new Error('Explain request failed. Is the backend running on port 8000?')
      const data = await res.json()
      setExplanation(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setExplainLoading(false)
    }
  }

  const selectedClassName = predictions.find((p) => p.class_idx === selectedIdx)?.class_name

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-6 py-5">
          <h1 className="text-2xl font-bold text-slate-800">
            ResNet18 · Explainable Image Classification
          </h1>
          <p className="text-slate-500 text-sm mt-1">
            Upload an image → get ImageNet-1K predictions from a pretrained ResNet18 → visualize
            Integrated Gradients attributions for any predicted class.
          </p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column: input + controls */}
        <div className="lg:col-span-1 space-y-6">
          <UploadCard onImageSelected={handleImageSelected} previewUrl={previewUrl} />

          <button
            onClick={handlePredict}
            disabled={!file || predictLoading}
            className="w-full py-2.5 rounded-lg bg-slate-800 hover:bg-slate-900 disabled:bg-slate-300 text-white font-medium transition-colors"
          >
            {predictLoading ? 'Predicting…' : 'Predict'}
          </button>

          <div className="bg-white rounded-2xl border border-slate-200 p-5">
            <h2 className="font-semibold text-slate-800 mb-3">Top-5 Predictions</h2>
            <PredictionsPanel
              predictions={predictions}
              selectedIdx={selectedIdx}
              onSelect={setSelectedIdx}
              loading={predictLoading}
            />
          </div>

          <ExplainControls
            baseline={baseline}
            setBaseline={setBaseline}
            steps={steps}
            setSteps={setSteps}
            onExplain={handleExplain}
            disabled={!file || selectedIdx === null}
            loading={explainLoading}
          />

          {error && (
            <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg p-3">
              {error}
            </div>
          )}
        </div>

        {/* Right column: explanation output */}
        <div className="lg:col-span-2 space-y-6">
          <ExplanationView explanation={explanation} className={selectedClassName} loading={explainLoading} />

          <div className="bg-white rounded-2xl border border-slate-200 p-5 text-sm text-slate-600 leading-relaxed">
            <h3 className="font-semibold text-slate-800 mb-2">How to read this</h3>
            <p className="mb-2">
              <b>Integrated Gradients</b> attributes the model's output for the selected class to
              each input pixel by accumulating gradients along a straight-line path from a{' '}
              <b>baseline</b> image to the actual input. Brighter / warmer regions in the heatmap
              contributed more strongly to the prediction.
            </p>
            <p>
              Try switching the baseline (black, white, noise, blurred) — Integrated Gradients
              attributions are baseline-dependent, and comparing them helps you judge how robust
              the explanation is, and whether the model is really focusing on the object rather
              than the background.
            </p>
          </div>
        </div>
      </main>

      <footer className="text-center text-xs text-slate-400 py-6">
        Experiment 5 · ResNet18 (ImageNet-1K) + Integrated Gradients (Captum)
      </footer>
    </div>
  )
}
