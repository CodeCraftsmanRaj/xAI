import React, { useRef, useState } from 'react'

export default function UploadCard({ onImageSelected, previewUrl }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFiles = (files) => {
    if (files && files.length > 0) {
      onImageSelected(files[0])
    }
  }

  return (
    <div
      className={`rounded-2xl border-2 border-dashed transition-colors p-6 text-center cursor-pointer bg-white
        ${dragOver ? 'border-brand-500 bg-brand-50' : 'border-slate-300 hover:border-brand-400'}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragOver(false)
        handleFiles(e.dataTransfer.files)
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      {previewUrl ? (
        <img src={previewUrl} alt="preview" className="mx-auto max-h-64 rounded-xl object-contain" />
      ) : (
        <div className="py-10 text-slate-500">
          <p className="text-lg font-medium">Drop an image here or click to upload</p>
          <p className="text-sm mt-1">Any JPG / PNG photo works</p>
        </div>
      )}
    </div>
  )
}
