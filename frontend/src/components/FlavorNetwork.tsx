import { useState, useEffect } from 'react'
import { NetworkData } from '../types'

interface FlavorNetworkProps {
  selectedIngredient: number | null
  onSelectIngredient: (id: number) => void
}

const CATEGORY_COLORS: Record<string, string> = {
  plant: '#4CAF50',
  animal: '#F44336',
  spice: '#FF9800',
  herb: '#8BC34A',
  dairy: '#2196F3',
  fruit: '#E91E63',
  vegetable: '#009688',
  cereal: '#795548',
  beverage: '#9C27B0',
  unknown: '#607D8B',
}

function getColor(category: string): string {
  return CATEGORY_COLORS[category.toLowerCase()] || CATEGORY_COLORS.unknown
}

function FlavorNetwork({ selectedIngredient, onSelectIngredient }: FlavorNetworkProps) {
  const [data, setData] = useState<NetworkData>({ nodes: [], edges: [] })
  const [minPmi, setMinPmi] = useState(2.0)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    const params = new URLSearchParams({
      min_pmi: String(minPmi),
      limit: '150',
    })
    if (selectedIngredient !== null) {
      params.set('ingredient', String(selectedIngredient))
    }
    fetch(`/api/network?${params}`)
      .then(r => r.json())
      .then((d: NetworkData) => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [selectedIngredient, minPmi])

  // Simple force-free layout: place nodes in a circle
  const cx = 400, cy = 300, radius = 240
  const positions = new Map<number, { x: number; y: number }>()
  data.nodes.forEach((n, i) => {
    const angle = (2 * Math.PI * i) / data.nodes.length
    positions.set(n.id, {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    })
  })

  const maxPmi = Math.max(...data.edges.map(e => e.pmi), 1)

  return (
    <div className="flavor-network">
      <div className="fn-controls">
        <label>
          Min PMI:
          <input
            type="range"
            min={0}
            max={5}
            step={0.1}
            value={minPmi}
            onChange={e => setMinPmi(parseFloat(e.target.value))}
          />
          <span>{minPmi.toFixed(1)}</span>
        </label>
        <span className="fn-stats">{data.nodes.length} nodes, {data.edges.length} edges</span>
      </div>

      {loading ? (
        <div className="fn-loading">Loading flavor network…</div>
      ) : (
        <svg viewBox="0 0 800 600" className="fn-svg">
          {data.edges.map((e, i) => {
            const s = positions.get(e.source)
            const t = positions.get(e.target)
            if (!s || !t) return null
            const width = Math.max(0.5, (e.pmi / maxPmi) * 4)
            return (
              <line
                key={i}
                x1={s.x} y1={s.y}
                x2={t.x} y2={t.y}
                stroke="#ccc"
                strokeWidth={width}
                strokeOpacity={0.6}
              />
            )
          })}
          {data.nodes.map(n => {
            const pos = positions.get(n.id)
            if (!pos) return null
            const isSelected = n.id === selectedIngredient
            return (
              <g key={n.id} onClick={() => onSelectIngredient(n.id)} style={{ cursor: 'pointer' }}>
                <circle
                  cx={pos.x} cy={pos.y}
                  r={isSelected ? 10 : 6}
                  fill={getColor(n.category)}
                  stroke={isSelected ? '#000' : 'none'}
                  strokeWidth={2}
                />
                <text
                  x={pos.x} y={pos.y - 12}
                  textAnchor="middle"
                  fontSize={isSelected ? 11 : 9}
                  fontWeight={isSelected ? 700 : 400}
                  fill="#333"
                >
                  {n.name}
                </text>
              </g>
            )
          })}
        </svg>
      )}

      <div className="fn-legend">
        {Object.entries(CATEGORY_COLORS).filter(([k]) => k !== 'unknown').map(([cat, color]) => (
          <span key={cat} className="fn-legend-item">
            <span className="fn-legend-dot" style={{ background: color }} />
            {cat}
          </span>
        ))}
      </div>
    </div>
  )
}

export default FlavorNetwork
