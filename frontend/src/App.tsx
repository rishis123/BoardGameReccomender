import { FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  GameSuggestion,
  QueryDimension,
  RagResponse,
  RecommendationResult,
} from './types'

type Method = 'svd' | 'tfidf'

function QueryDimsPanel({ dims, variant }: { dims: QueryDimension[]; variant?: 'ai' }) {
  if (dims.length === 0) return null
  const maxAbs = Math.max(...dims.map((d) => Math.abs(d.activation)), 0.001)
  return (
    <div className={`bg-qdims${variant === 'ai' ? ' bg-qdims--ai' : ''}`}>
      <div className="bg-qdims-title">Top activated dimensions</div>
      {dims.map((d) => {
        const pct = Math.round((Math.abs(d.activation) / maxAbs) * 100)
        return (
          <div key={d.index} className="bg-qdim-row">
            <div className="bg-qdim-bar-wrap">
              <div
                className={`bg-qdim-bar${d.activation < 0 ? ' neg' : ''}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="bg-qdim-info">
              <span className="bg-qdim-label">D{d.index + 1}: {d.label}</span>
              <span className="bg-qdim-score">{d.activation > 0 ? '+' : ''}{d.activation.toFixed(3)}</span>
            </div>
            <div className="bg-tags" style={{ marginTop: 4 }}>
              {d.terms.slice(0, 4).map((t, i) => (
                <span key={i} className="tag muted">{t.term}</span>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function GameThumbnail({ url, name }: { url: string | null; name: string }) {
  const [failed, setFailed] = useState(false)
  const initials = name.split(' ').slice(0, 2).map((w) => w[0]).join('').toUpperCase()

  if (!url || failed) {
    return <div className="bg-thumb bg-thumb--fallback">{initials}</div>
  }
  return (
    <img
      className="bg-thumb"
      src={url}
      alt={name}
      onError={() => setFailed(true)}
    />
  )
}

function GameCard({ r }: { r: RecommendationResult }) {
  return (
    <article className="bg-card">
      <div className="bg-card-head">
        <GameThumbnail url={r.thumbnail ?? null} name={r.name} />
        <div className="bg-card-head-text">
          <div>
            <h3>
              <a
                href={`https://www.google.com/search?q=${encodeURIComponent(r.name + ' board game')}`}
                target="_blank"
                rel="noreferrer"
              >{r.name}</a>
            </h3>
            <div className="bg-subtle">Published: {r.year_published || '—'}</div>
          </div>
          <div className="bg-ranks">SVD #{r.rank_svd} · TF-IDF #{r.rank_tfidf}</div>
        </div>
      </div>

      <p>{r.snippet || 'No description snippet available.'}</p>

      <div className="bg-tags">
        {r.category && <span className="tag muted">{r.category}</span>}
        {r.mechanic && <span className="tag muted">{r.mechanic}</span>}
      </div>

      <div className="bg-meta">
        <span>Avg: {r.average_rating?.toFixed?.(2) ?? '—'}</span>
        <span>Ratings: {r.users_rated}</span>
        <span>SVD: {r.score_svd.toFixed(4)}</span>
        <span>TF-IDF: {r.score_tfidf.toFixed(4)}</span>
      </div>

      <div className="bg-tags">
        {r.why_tags.map((t) => (
          <span key={`${r.id}-${t.index}`} className="tag">
            Why: {t.label} ({t.activation})
          </span>
        ))}
      </div>
    </article>
  )
}

function App(): JSX.Element {
  // Field 1: seed game search
  const [seedQuery, setSeedQuery] = useState('')
  const [suggestions, setSuggestions] = useState<GameSuggestion[]>([])
  const [seed, setSeed] = useState<GameSuggestion | null>(null)
  const [seedDims, setSeedDims] = useState<QueryDimension[]>([])

  // Field 2: clarifying details / standalone text query
  const [details, setDetails] = useState('')

  const [method, setMethod] = useState<Method>('svd')
  const [k, setK] = useState(8)

  // Results
  const [results, setResults] = useState<RecommendationResult[]>([])
  const [ragResults, setRagResults] = useState<RecommendationResult[]>([])
  const [originalLabel, setOriginalLabel] = useState('')
  const [rewrittenQuery, setRewrittenQuery] = useState('')
  const [originalDims, setOriginalDims] = useState<QueryDimension[]>([])
  const [rewrittenDims, setRewrittenDims] = useState<QueryDimension[]>([])
  const [ragError, setRagError] = useState<string | null>(null)
  const [hasResults, setHasResults] = useState(false)

  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('Pick a game, describe what you want, or both.')
  const controlsRef = useRef<HTMLElement | null>(null)

  // Autocomplete for seed field
  useEffect(() => {
    const timer = setTimeout(async () => {
      const q = seedQuery.trim()
      if (q.length < 2 || seed) { setSuggestions([]); return }
      try {
        const res = await fetch(`/api/games/search?q=${encodeURIComponent(q)}`)
        setSuggestions(await res.json())
      } catch { setSuggestions([]) }
    }, 200)
    return () => clearTimeout(timer)
  }, [seedQuery, seed])

  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (!controlsRef.current?.contains(e.target as Node)) setSuggestions([])
    }
    document.addEventListener('mousedown', handleOutsideClick)
    return () => document.removeEventListener('mousedown', handleOutsideClick)
  }, [])

  const canSearch = useMemo(
    () => seed !== null || details.trim().length > 1,
    [seed, details]
  )

  const selectSeed = async (game: GameSuggestion) => {
    setSeed(game)
    setSeedQuery(game.name)
    setSuggestions([])
    try {
      const res = await fetch(`/api/games/dimensions?id=${game.id}`)
      setSeedDims(await res.json())
    } catch { setSeedDims([]) }
  }

  const clearSeed = () => {
    setSeed(null)
    setSeedQuery('')
    setSeedDims([])
  }

  const runRecommendation = async () => {
    if (!canSearch || loading) return
    setLoading(true)
    setMessage('Running retrieval...')

    const params = new URLSearchParams({ method, k: String(k) })
    if (seed) params.set('seed', seed.id)
    if (details.trim()) params.set('q', details.trim())

    try {
      const res = await fetch(`/api/rag?${params}`)
      const data: RagResponse = await res.json()
      setResults(data.original_results || [])
      setRagResults(data.rag_results || [])
      setOriginalLabel(data.original_label || '')
      setRewrittenQuery(data.rewritten_query || '')
      setOriginalDims(data.original_dims || [])
      setRewrittenDims(data.rewritten_dims || [])
      setRagError(data.error || null)
      setHasResults(true)
      setMessage(`Showing ${data.original_results?.length ?? 0} results per column.`)
    } catch {
      setResults([])
      setRagResults([])
      setMessage('Request failed. Check backend server logs.')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault(); setSuggestions([]); await runRecommendation()
  }

  const renderCards = (items: RecommendationResult[]) =>
    items.length === 0
      ? <div className="bg-empty">No results.</div>
      : items.map((r) => <GameCard key={r.id} r={r} />)

  return (
    <div className="bg-app">
      <header className="bg-header">
        <h1>Board Game Recommender</h1>
        <p>Find similar games using TF-IDF and latent SVD themes with clear explanations.</p>
      </header>

      <section className="bg-controls" ref={controlsRef}>
        <div className="bg-fields">
          {/* Field 1: seed game */}
          <div className="bg-field">
            <label className="bg-field-label">Similar to</label>
            {seed ? (
              <div className="bg-seed-locked">
                <span className="bg-seed-name">{seed.name}</span>
                <span className="bg-seed-meta">{seed.year_published || '—'} · {seed.users_rated} ratings</span>
                <button type="button" className="bg-seed-clear" onClick={clearSeed}>✕</button>
              </div>
            ) : (
              <div className="bg-input-wrap">
                <input
                  value={seedQuery}
                  onChange={(e) => setSeedQuery(e.target.value)}
                  placeholder="Search a game title…"
                  autoComplete="off"
                  onFocus={() => {
                    if (seedQuery.trim().length > 1) {
                      void (async () => {
                        try {
                          const res = await fetch(`/api/games/search?q=${encodeURIComponent(seedQuery.trim())}`)
                          setSuggestions(await res.json())
                        } catch { setSuggestions([]) }
                      })()
                    }
                  }}
                />
                {suggestions.length > 0 && (
                  <ul className="bg-suggestions">
                    {suggestions.map((s) => (
                      <li key={s.id} onMouseDown={(e) => e.preventDefault()} onClick={() => void selectSeed(s)}>
                        <span>{s.name}</span>
                        <small>{s.year_published || '—'} · {s.users_rated} ratings</small>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
            {/* Inline seed dims preview */}
            {seedDims.length > 0 && (
              <div className="bg-seed-dims">
                {seedDims.slice(0, 4).map((d) => (
                  <span key={d.index} className="bg-seed-dim-tag">
                    {d.label} <em>{d.activation > 0 ? '+' : ''}{d.activation.toFixed(2)}</em>
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Field 2: clarifying details */}
          <div className="bg-field">
            <label className="bg-field-label">Looking for</label>
            <input
              value={details}
              onChange={(e) => setDetails(e.target.value)}
              placeholder={seed ? `e.g. shorter play time, more player interaction, fantasy theme…` : `e.g. strategic hex game with resource trading…`}
              autoComplete="off"
              onKeyDown={(e) => {
                if (e.key === 'Enter') { e.preventDefault(); setSuggestions([]); void runRecommendation() }
              }}
            />
          </div>
        </div>

        <form className="bg-row" onSubmit={handleSubmit}>
          <label>
            Method
            <select value={method} onChange={(e) => setMethod(e.target.value as Method)}>
              <option value="svd">SVD (latent)</option>
              <option value="tfidf">TF-IDF (baseline)</option>
            </select>
          </label>

          <label>
            Top K
            <input
              type="number" min={1} max={20} value={k}
              onChange={(e) => setK(Math.max(1, Math.min(20, Number(e.target.value) || 8)))}
            />
          </label>

          <button type="submit" disabled={!canSearch || loading}>
            {loading ? 'Searching...' : 'Recommend'}
          </button>
        </form>
      </section>

      <p className="bg-message">{message}</p>

      {hasResults && (
        <main className="bg-main">
          <div className="bg-two-cols">
            {/* Left: standard IR */}
            <section className="bg-col">
              <div className="bg-section-head">
                <h2>Standard Results</h2>
                <span>{results.length} items</span>
              </div>
              <div className="bg-query-pill">
                <span className="bg-query-label">Query</span>
                {originalLabel}
              </div>
              <QueryDimsPanel dims={originalDims} />
              <div className="bg-col-cards">{renderCards(results)}</div>
            </section>

            {/* Right: AI-enhanced IR */}
            <section className="bg-col bg-col--ai">
              <div className="bg-section-head">
                <h2>AI-Enhanced Results</h2>
                <span>{ragResults.length} items</span>
              </div>
              {ragError ? (
                <div className="bg-empty bg-rag-error">{ragError}</div>
              ) : (
                <>
                  <div className="bg-query-pill bg-query-pill--ai">
                    <span className="bg-query-label">AI rewrote to</span>
                    {rewrittenQuery}
                  </div>
                  <QueryDimsPanel dims={rewrittenDims} variant="ai" />
                  <div className="bg-col-cards">{renderCards(ragResults)}</div>
                </>
              )}
            </section>
          </div>
        </main>
      )}
    </div>
  )
}

export default App
