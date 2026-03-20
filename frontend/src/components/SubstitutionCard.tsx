import { Substitution } from '../types'

interface SubstitutionCardProps {
  sub: Substitution
  onSelect: (id: number) => void
}

function SubstitutionCard({ sub, onSelect }: SubstitutionCardProps) {
  const pct = Math.round(sub.similarity * 100)

  return (
    <div className="sub-card" onClick={() => onSelect(sub.id)}>
      <div className="sub-card-header">
        <span className="sub-card-name">{sub.name}</span>
        <span className="sub-card-score">{pct}%</span>
      </div>
      <span className="sub-card-category">{sub.category}</span>
      {sub.shared_molecules.length > 0 && (
        <div className="sub-card-molecules">
          {sub.shared_molecules.slice(0, 4).map((m, i) => (
            <span key={i} className="molecule-tag">
              {m.common_name || m.pubchem_id}
            </span>
          ))}
          {sub.shared_molecules.length > 4 && (
            <span className="molecule-tag more">+{sub.shared_molecules.length - 4}</span>
          )}
        </div>
      )}
    </div>
  )
}

export default SubstitutionCard
