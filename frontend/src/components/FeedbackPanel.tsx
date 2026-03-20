import { useState, useEffect } from 'react'
import { MetricsData } from '../types'

function FeedbackPanel() {
  const [metrics, setMetrics] = useState<MetricsData>({
    avg_feedback: 0,
    total_feedback: 0,
  })

  const fetchMetrics = () => {
    fetch('/api/metrics')
      .then(r => r.json())
      .then((data: MetricsData) => setMetrics(data))
      .catch(() => {})
  }

  useEffect(() => {
    fetchMetrics()
    const interval = setInterval(fetchMetrics, 30000)
    return () => clearInterval(interval)
  }, [])

  const avgPct = metrics.avg_feedback
    ? ((metrics.avg_feedback + 1) / 2 * 100).toFixed(0)
    : '—'

  return (
    <div className="feedback-panel">
      <h4>Performance Metrics</h4>
      <div className="fp-grid">
        <div className="fp-metric">
          <span className="fp-value">{avgPct}{avgPct !== '—' ? '%' : ''}</span>
          <span className="fp-label">Avg Satisfaction</span>
        </div>
        <div className="fp-metric">
          <span className="fp-value">{metrics.total_feedback}</span>
          <span className="fp-label">Total Ratings</span>
        </div>
        {metrics.precision_at_k !== undefined && (
          <div className="fp-metric">
            <span className="fp-value">{(metrics.precision_at_k * 100).toFixed(1)}%</span>
            <span className="fp-label">Precision@10</span>
          </div>
        )}
      </div>
    </div>
  )
}

export default FeedbackPanel
