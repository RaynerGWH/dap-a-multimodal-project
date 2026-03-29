import { useState, useRef } from 'react'

const MODEL_INFO = {
  bert: {
    name: 'BERT',
    desc: 'Text only (headline)',
    color: '#888',
    tag: 'Baseline',
  },
  vit: {
    name: 'ViT',
    desc: 'Image only',
    color: '#888',
    tag: 'Baseline',
  },
  cross_attention: {
    name: 'Cross-attention',
    desc: 'BERT + ViT fusion',
    color: '#E8593C',
    tag: 'Task-specific',
  },
  qwen3_zero_shot: {
    name: 'Qwen3-VL',
    desc: 'Zero-shot (no training)',
    color: '#7F77DD',
    tag: 'Zero-shot',
  },
  qwen3_finetuned: {
    name: 'Qwen3-VL LoRA',
    desc: 'Fine-tuned on N24News',
    color: '#1D9E75',
    tag: 'Fine-tuned',
  },
}

function ProbBar({ category, prob, isTop }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8, fontSize: 12,
      opacity: isTop ? 1 : 0.5,
    }}>
      <span style={{
        width: 120, textAlign: 'right', fontFamily: "'DM Sans', sans-serif",
        fontWeight: isTop ? 700 : 400, whiteSpace: 'nowrap', overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}>{category}</span>
      <div style={{
        flex: 1, height: 18, background: 'rgba(255,255,255,0.06)',
        borderRadius: 4, overflow: 'hidden',
      }}>
        <div style={{
          width: `${(prob * 100).toFixed(1)}%`, height: '100%',
          background: isTop ? 'var(--accent)' : 'rgba(255,255,255,0.15)',
          borderRadius: 4, transition: 'width 0.6s ease',
        }} />
      </div>
      <span style={{
        width: 48, fontFamily: "'JetBrains Mono', monospace", fontSize: 11,
        color: isTop ? 'var(--accent)' : '#888',
      }}>{(prob * 100).toFixed(1)}%</span>
    </div>
  )
}

function ModelCard({ modelKey, result, loading }) {
  const info = MODEL_INFO[modelKey]
  const hasProbs = result?.probabilities

  // Sort probabilities and take top 5
  let topProbs = []
  if (hasProbs) {
    topProbs = Object.entries(result.probabilities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
  }

  return (
    <div style={{
      '--accent': info.color,
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: 16, padding: 24, flex: 1, minWidth: 280,
      position: 'relative', overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 3,
        background: info.color,
      }} />

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: '#fff' }}>{info.name}</h3>
          <p style={{ margin: '4px 0 0', fontSize: 13, color: '#888' }}>{info.desc}</p>
        </div>
        <span style={{
          fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1,
          padding: '4px 10px', borderRadius: 100, background: `${info.color}22`,
          color: info.color,
        }}>{info.tag}</span>
      </div>

      {loading ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#888', fontSize: 14 }}>
          <div className="spinner" /> Classifying...
        </div>
      ) : result ? (
        <div>
          <div style={{
            fontSize: 22, fontWeight: 700, color: info.color,
            marginBottom: hasProbs ? 16 : 8,
            fontFamily: "'DM Sans', sans-serif",
          }}>
            {result.prediction}
          </div>

          {hasProbs && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {topProbs.map(([cat, prob], i) => (
                <ProbBar key={cat} category={cat} prob={prob} isTop={i === 0} />
              ))}
            </div>
          )}

          {result.raw_output && !hasProbs && (
            <p style={{ fontSize: 12, color: '#666', margin: '8px 0 0', fontFamily: "'JetBrains Mono', monospace" }}>
              Raw: "{result.raw_output}"
            </p>
          )}

          {result.error && (
            <p style={{ fontSize: 12, color: '#E8593C', margin: '8px 0 0' }}>
              Error: {result.error}
            </p>
          )}
        </div>
      ) : (
        <p style={{ color: '#555', fontSize: 14, margin: 0 }}>Upload an image and enter a headline to classify</p>
      )}
    </div>
  )
}

export default function App() {
  const [apiUrl, setApiUrl] = useState(localStorage.getItem('apiUrl') || '')
  const [headline, setHeadline] = useState('')
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const fileRef = useRef()

  const handleApiUrl = (url) => {
    setApiUrl(url)
    localStorage.setItem('apiUrl', url)
  }

  const handleImage = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImage(file)
      setImagePreview(URL.createObjectURL(file))
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      setImage(file)
      setImagePreview(URL.createObjectURL(file))
    }
  }

  const classify = async () => {
    if (!apiUrl || !image || !headline) {
      setError('Need API URL, image, and headline')
      return
    }
    setError('')
    setLoading(true)
    setResults(null)

    try {
      const formData = new FormData()
      formData.append('file', image)
      formData.append('headline', headline)

      const url = apiUrl.endsWith('/') ? apiUrl : apiUrl + '/'
      const res = await fetch(`${url}predict`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setResults(data)
    } catch (e) {
      setError(`Failed to connect: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !loading) classify()
  }

  return (
    <div style={{
      minHeight: '100vh', background: '#0a0a0b', color: '#fff',
      fontFamily: "'DM Sans', sans-serif",
    }}>
      {/* Header */}
      <div style={{
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        padding: '20px 32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, letterSpacing: -0.5 }}>
            Multimodal News Classifier
          </h1>
          <p style={{ margin: '2px 0 0', fontSize: 12, color: '#666' }}>
            Cross-attention vs Qwen3-VL — DAP Hack Day
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: '#555' }}>API:</span>
          <input
            type="text"
            placeholder="https://your-pod-url:8000"
            value={apiUrl}
            onChange={(e) => handleApiUrl(e.target.value)}
            style={{
              width: 280, padding: '8px 12px', fontSize: 12, borderRadius: 8,
              border: '1px solid rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.04)',
              color: '#fff', outline: 'none', fontFamily: "'JetBrains Mono', monospace",
            }}
          />
        </div>
      </div>

      {/* Main content */}
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '32px 24px' }}>
        {/* Input section */}
        <div style={{ display: 'flex', gap: 24, marginBottom: 32 }}>
          {/* Image upload */}
          <div
            onClick={() => fileRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            style={{
              width: 280, minHeight: 200, borderRadius: 16,
              border: '2px dashed rgba(255,255,255,0.1)',
              background: 'rgba(255,255,255,0.02)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'pointer', overflow: 'hidden', flexShrink: 0,
              transition: 'border-color 0.2s',
            }}
            onMouseEnter={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.25)'}
            onMouseLeave={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'}
          >
            {imagePreview ? (
              <img src={imagePreview} alt="Preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            ) : (
              <div style={{ textAlign: 'center', color: '#555', padding: 24 }}>
                <div style={{ fontSize: 36, marginBottom: 8 }}>+</div>
                <div style={{ fontSize: 13 }}>Drop image or click to upload</div>
              </div>
            )}
            <input ref={fileRef} type="file" accept="image/*" onChange={handleImage} style={{ display: 'none' }} />
          </div>

          {/* Headline + classify */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div>
              <label style={{ fontSize: 12, color: '#888', marginBottom: 6, display: 'block', fontWeight: 500 }}>
                News headline
              </label>
              <input
                type="text"
                placeholder="Enter a news headline..."
                value={headline}
                onChange={(e) => setHeadline(e.target.value)}
                onKeyDown={handleKeyDown}
                style={{
                  width: '100%', padding: '14px 18px', fontSize: 16, borderRadius: 12,
                  border: '1px solid rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.04)',
                  color: '#fff', outline: 'none', fontFamily: "'DM Sans', sans-serif",
                  boxSizing: 'border-box',
                }}
              />
            </div>

            <button
              onClick={classify}
              disabled={loading || !image || !headline || !apiUrl}
              style={{
                padding: '14px 32px', fontSize: 14, fontWeight: 700, borderRadius: 12,
                border: 'none', cursor: loading ? 'wait' : 'pointer',
                background: loading ? '#333' : '#fff', color: loading ? '#888' : '#0a0a0b',
                transition: 'all 0.2s', fontFamily: "'DM Sans', sans-serif",
                opacity: (!image || !headline || !apiUrl) ? 0.3 : 1,
              }}
            >
              {loading ? 'Classifying...' : 'Classify'}
            </button>

            {error && (
              <p style={{ color: '#E8593C', fontSize: 13, margin: 0 }}>{error}</p>
            )}
          </div>
        </div>

        {/* Results */}
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          {Object.keys(MODEL_INFO).map((key) => (
            <ModelCard
              key={key}
              modelKey={key}
              result={results?.[key]}
              loading={loading}
            />
          ))}
        </div>

        {/* Accuracy summary */}
        <div style={{
          marginTop: 32, padding: 24, borderRadius: 16,
          background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
        }}>
          <h3 style={{ margin: '0 0 16px', fontSize: 14, fontWeight: 500, color: '#888' }}>
            Model performance on N24News (val set)
          </h3>
          <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap' }}>
            {[
              { name: 'BERT (text only)', acc: '73.6%', color: '#888' },
              { name: 'ViT (image only)', acc: '53.9%', color: '#888' },
              { name: 'Cross-attention', acc: '78.9%', color: '#E8593C' },
              { name: 'Qwen3-VL zero-shot', acc: '57.6%', color: '#7F77DD' },
              { name: 'Qwen3-VL fine-tuned', acc: '60.3%', color: '#1D9E75' },
            ].map((m) => (
              <div key={m.name} style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: m.color, fontFamily: "'JetBrains Mono', monospace" }}>
                  {m.acc}
                </div>
                <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>{m.name}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        .spinner {
          width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.1);
          border-top-color: #fff; border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        input::placeholder { color: #444; }
        * { box-sizing: border-box; }
      `}</style>
    </div>
  )
}
