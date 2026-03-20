import { useState, useEffect } from 'react'
import { IngredientProfile } from '../types'

interface IngredientProfileTableProps {
  ingredientId: number | null
  onClose: () => void
}

function IngredientProfileTable({ ingredientId, onClose }: IngredientProfileTableProps) {
  const [profile, setProfile] = useState<IngredientProfile | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (ingredientId === null) { setProfile(null); return }
    setLoading(true)
    fetch(`/api/ingredient-profile?id=${ingredientId}`)
      .then(r => r.json())
      .then((data: IngredientProfile) => {
        if (data.id) setProfile(data)
        else setProfile(null)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [ingredientId])

  if (ingredientId === null) return null

  return (
    <div className="profile-panel">
      <div className="profile-header">
        <h3>{loading ? 'Loading…' : profile?.name || 'Unknown'}</h3>
        <button className="profile-close" onClick={onClose}>x</button>
      </div>
      {profile && (
        <>
          <div className="profile-meta">
            <span>Category: {profile.category}</span>
            {profile.scientific_name && <span>Scientific: {profile.scientific_name}</span>}
            <span>{profile.molecule_count} molecules</span>
          </div>
          <div className="profile-molecules">
            <table>
              <thead>
                <tr>
                  <th>Molecule</th>
                  <th>PubChem ID</th>
                  <th>Flavor Profile</th>
                </tr>
              </thead>
              <tbody>
                {profile.molecules.map(m => (
                  <tr key={m.id}>
                    <td>{m.common_name || '—'}</td>
                    <td>{m.pubchem_id}</td>
                    <td>{m.flavor_profile || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}

export default IngredientProfileTable
