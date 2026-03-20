import { useState, useEffect, useRef } from 'react'
import { SensoryMapData, SensoryPoint } from '../types'

interface SensoryMap3DProps {
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

function SensoryMap3D({ onSelectIngredient }: SensoryMap3DProps) {
  const [data, setData] = useState<SensoryMapData>({ points: [], dimensions: [] })
  const [dims, setDims] = useState([0, 1, 2])
  const [category, setCategory] = useState('')
  const [loading, setLoading] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setLoading(true)
    const params = new URLSearchParams({
      dims: dims.join(','),
    })
    if (category) params.set('category', category)
    fetch(`/api/sensory-map?${params}`)
      .then(r => r.json())
      .then((d: SensoryMapData) => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [dims, category])

  useEffect(() => {
    if (!containerRef.current || data.points.length === 0) return

    const loadPlotly = async () => {
      // @ts-expect-error Plotly loaded via CDN
      const Plotly = window.Plotly
      if (!Plotly) return

      const groups: Record<string, SensoryPoint[]> = {}
      for (const p of data.points) {
        const cat = p.category || 'Unknown'
        if (!groups[cat]) groups[cat] = []
        groups[cat].push(p)
      }

      const traces = Object.entries(groups).map(([cat, points]) => ({
        type: 'scatter3d' as const,
        mode: 'markers' as const,
        name: cat,
        x: points.map(p => p.x),
        y: points.map(p => p.y),
        z: points.map(p => p.z),
        text: points.map(p => p.name),
        customdata: points.map(p => p.id),
        marker: {
          size: 4,
          color: CATEGORY_COLORS[cat.toLowerCase()] || CATEGORY_COLORS.unknown,
          opacity: 0.8,
        },
        hovertemplate: '%{text}<extra>%{fullData.name}</extra>',
      }))

      const dimLabels = data.dimensions.map(d => d.label)
      const layout = {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        scene: {
          xaxis: { title: dimLabels[0] || 'Dim 0' },
          yaxis: { title: dimLabels[1] || 'Dim 1' },
          zaxis: { title: dimLabels[2] || 'Dim 2' },
        },
        legend: { x: 0, y: 1 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
      }

      Plotly.newPlot(containerRef.current, traces, layout, { responsive: true })

      // Plotly attaches event methods to the DOM element at runtime
      const plotEl = containerRef.current as unknown as {
        on: (event: string, cb: (data: { points: { customdata: number }[] }) => void) => void
      }
      plotEl?.on('plotly_click', (eventData) => {
        if (eventData.points.length > 0) {
          onSelectIngredient(eventData.points[0].customdata)
        }
      })
    }

    loadPlotly()
  }, [data, onSelectIngredient])

  const categories = [...new Set(data.points.map(p => p.category))].sort()

  return (
    <div className="sensory-map">
      <div className="sm-controls">
        <label>
          Dim X:
          <input type="number" min={0} max={19} value={dims[0]}
            onChange={e => setDims([+e.target.value, dims[1], dims[2]])} />
        </label>
        <label>
          Dim Y:
          <input type="number" min={0} max={19} value={dims[1]}
            onChange={e => setDims([dims[0], +e.target.value, dims[2]])} />
        </label>
        <label>
          Dim Z:
          <input type="number" min={0} max={19} value={dims[2]}
            onChange={e => setDims([dims[0], dims[1], +e.target.value])} />
        </label>
        <select value={category} onChange={e => setCategory(e.target.value)}>
          <option value="">All Categories</option>
          {categories.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      {loading ? (
        <div className="sm-loading">Loading sensory map…</div>
      ) : (
        <div ref={containerRef} className="sm-plot" />
      )}

      {data.dimensions.length > 0 && (
        <div className="sm-dim-info">
          {data.dimensions.map(d => (
            <div key={d.index} className="sm-dim-card">
              <strong>{d.label}</strong>
              <span className="sm-var">({(d.explained_variance * 100).toFixed(1)}% var)</span>
              <div className="sm-dim-mols">
                {d.top_molecules.slice(0, 5).map((m, i) => (
                  <span key={i} className="molecule-tag">{m.name || `mol_${m.molecule_id}`}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default SensoryMap3D
