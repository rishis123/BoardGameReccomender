import { useState, useEffect, useCallback } from 'react'
import { Ingredient, Substitution } from '../types'
import SubstitutionCard from './SubstitutionCard'

interface SubstitutionEngineProps {
  onSelectIngredient: (id: number) => void
}

function SubstitutionEngine({ onSelectIngredient }: SubstitutionEngineProps) {
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState<Ingredient[]>([])
  const [seedId, setSeedId] = useState<number | null>(null)
  const [seedName, setSeedName] = useState('')
  const [results, setResults] = useState<Substitution[]>([])
  const [category, setCategory] = useState('')
  const [loading, setLoading] = useState(false)

  const searchIngredients = useCallback(async (q: string) => {
    if (q.length < 2) { setSuggestions([]); return }
    const resp = await fetch(`/api/ingredients/search?q=${encodeURIComponent(q)}`)
    const data: Ingredient[] = await resp.json()
    setSuggestions(data)
  }, [])

  useEffect(() => {
    const timer = setTimeout(() => searchIngredients(query), 250)
    return () => clearTimeout(timer)
  }, [query, searchIngredients])

  const selectSeed = async (ing: Ingredient) => {
    setSeedId(ing.id)
    setSeedName(ing.name)
    setQuery(ing.name)
    setSuggestions([])
    await fetchSubstitutes(ing.id, category)
  }

  const fetchSubstitutes = async (id: number, cat: string) => {
    setLoading(true)
    const params = new URLSearchParams({ seed: String(id), k: '20' })
    if (cat) params.set('category', cat)
    const resp = await fetch(`/api/substitutions?${params}`)
    const data: Substitution[] = await resp.json()
    setResults(data)
    setLoading(false)
  }

  const handleCategoryChange = (cat: string) => {
    setCategory(cat)
    if (seedId !== null) fetchSubstitutes(seedId, cat)
  }

  const categories = [...new Set(results.map(r => r.category))].sort()

  return (
    <div className="substitution-engine">
      <div className="se-search-wrapper">
        <input
          className="se-search"
          placeholder="Enter an ingredient (e.g. Strawberry)"
          value={query}
          onChange={e => setQuery(e.target.value)}
          autoComplete="off"
        />
        {suggestions.length > 0 && (
          <ul className="se-suggestions">
            {suggestions.map(s => (
              <li key={s.id} onClick={() => selectSeed(s)}>
                <span className="sugg-name">{s.name}</span>
                <span className="sugg-cat">{s.category}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {seedName && (
        <div className="se-filter-row">
          <span className="se-seed-label">Substitutes for <strong>{seedName}</strong></span>
          <select
            className="se-category-select"
            value={category}
            onChange={e => handleCategoryChange(e.target.value)}
          >
            <option value="">All Categories</option>
            {categories.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
      )}

      {loading && <div className="se-loading">Searching molecular profiles…</div>}

      <div className="se-results">
        {results.map(sub => (
          <SubstitutionCard
            key={sub.id}
            sub={sub}
            onSelect={onSelectIngredient}
          />
        ))}
      </div>
    </div>
  )
}

export default SubstitutionEngine
