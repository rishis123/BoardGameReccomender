import { useEffect, useMemo, useState } from 'react'
import './App.css'
import { GameSuggestion, RecommendationResult, RecommendationResponse } from './types'

type Method = 'svd' | 'tfidf'

function App(): JSX.Element {
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState<GameSuggestion[]>([])
  const [seed, setSeed] = useState<GameSuggestion | null>(null)
  const [method, setMethod] = useState<Method>('svd')
  const [k, setK] = useState(8)
  const [results, setResults] = useState<RecommendationResult[]>([])
  const [latent, setLatent] = useState<RecommendationResponse['latent_dimensions']>([])
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('Choose a game title or enter a thematic query.')

  useEffect(() => {
    const timer = setTimeout(async () => {
      const q = query.trim()
      if (q.length < 2 || seed) {
        setSuggestions([])
        return
      }

      try {
        const res = await fetch(`/api/games/search?q=${encodeURIComponent(q)}`)
        const data: GameSuggestion[] = await res.json()
        setSuggestions(data)
      } catch {
        setSuggestions([])
      }
    }, 200)

    return () => clearTimeout(timer)
  }, [query, seed])

  const canSearch = useMemo(() => {
    if (seed) return true
    return query.trim().length > 1
  }, [query, seed])

  const selectSeed = (game: GameSuggestion) => {
    setSeed(game)
    setQuery(game.name)
    setSuggestions([])
  }

  const clearSeed = () => {
    setSeed(null)
  }

  const runRecommendation = async () => {
    if (!canSearch || loading) return

    setLoading(true)
    setMessage('Running retrieval...')

    const params = new URLSearchParams({
      method,
      k: String(k),
    })

    if (seed) {
      params.set('seed', seed.id)
    } else {
      params.set('q', query.trim())
    }

    try {
      const res = await fetch(`/api/recommendations?${params.toString()}`)
      const data: RecommendationResponse = await res.json()
      setResults(data.recommendations || [])
      setLatent(data.latent_dimensions || [])

      if (!data.recommendations || data.recommendations.length === 0) {
        setMessage('No recommendations returned. Try a broader query.')
      } else {
        setMessage(`Showing top ${data.recommendations.length} recommendations.`)
      }
    } catch {
      setResults([])
      setMessage('Request failed. Check backend server logs.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-app">
      <header className="bg-header">
        <h1>Board Game Recommender</h1>
        <p>TF-IDF baseline + SVD latent themes with explainable tags.</p>
      </header>

      <section className="bg-controls">
        <div className="bg-input-wrap">
          <input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              if (seed) setSeed(null)
            }}
            placeholder="Search title or enter a query (e.g., strategic medieval war game with no dice)"
            autoComplete="off"
          />
          {suggestions.length > 0 && (
            <ul className="bg-suggestions">
              {suggestions.map((s) => (
                <li key={s.id} onClick={() => selectSeed(s)}>
                  <span>{s.name}</span>
                  <small>{s.year_published || '—'} · {s.users_rated} ratings</small>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="bg-row">
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
              type="number"
              min={1}
              max={20}
              value={k}
              onChange={(e) => setK(Math.max(1, Math.min(20, Number(e.target.value) || 8)))}
            />
          </label>

          <button onClick={runRecommendation} disabled={!canSearch || loading}>
            {loading ? 'Searching...' : 'Recommend'}
          </button>

          {seed && (
            <button className="secondary" onClick={clearSeed}>
              Clear title mode
            </button>
          )}
        </div>
      </section>

      <p className="bg-message">{message}</p>

      <main className="bg-main">
        <section className="bg-results">
          {results.map((r) => (
            <article key={r.id} className="bg-card">
              <div className="bg-card-head">
                <h3>{r.name}</h3>
                <span>{r.year_published || '—'}</span>
              </div>

              <p>{r.snippet || 'No description snippet available.'}</p>

              <div className="bg-meta">
                <span>Avg: {r.average_rating?.toFixed?.(2) ?? '—'}</span>
                <span>Ratings: {r.users_rated}</span>
                <span>SVD: {r.score_svd.toFixed(4)}</span>
                <span>TF-IDF: {r.score_tfidf.toFixed(4)}</span>
                <span>Ranks: SVD #{r.rank_svd} / TF-IDF #{r.rank_tfidf}</span>
              </div>

              <div className="bg-tags">
                {r.why_tags.map((t) => (
                  <span key={`${r.id}-${t.index}`} className="tag">
                    Why: {t.label} ({t.activation})
                  </span>
                ))}
              </div>
            </article>
          ))}
        </section>

        <aside className="bg-latent">
          <h2>Latent Dimensions</h2>
          {latent.slice(0, 10).map((d) => (
            <div key={d.index} className="bg-latent-card">
              <strong>D{d.index + 1}: {d.label}</strong>
              <div className="bg-latent-var">Explained variance: {(d.explained_variance * 100).toFixed(2)}%</div>
              <div className="bg-tags">
                {d.terms.slice(0, 6).map((t, i) => (
                  <span key={`${d.index}-${i}`} className="tag muted">{t.term}</span>
                ))}
              </div>
            </div>
          ))}
        </aside>
      </main>
    </div>
  )
}

export default App
