import { useState, useEffect } from 'react'
import './App.css'
import SubstitutionEngine from './components/SubstitutionEngine'
import FlavorNetwork from './components/FlavorNetwork'
import SensoryMap3D from './components/SensoryMap3D'
import IngredientProfileTable from './components/IngredientProfileTable'
import FeedbackPanel from './components/FeedbackPanel'
import Chat from './Chat'

type View = 'substitution' | 'network' | 'sensory'

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [view, setView] = useState<View>('substitution')
  const [selectedIngredient, setSelectedIngredient] = useState<number | null>(null)
  const [profileId, setProfileId] = useState<number | null>(null)
  const [chatOpen, setChatOpen] = useState(false)

  useEffect(() => {
    fetch('/api/config')
      .then(r => r.json())
      .then(data => setUseLlm(data.use_llm))
  }, [])

  const handleSelectIngredient = (id: number) => {
    setSelectedIngredient(id)
    setProfileId(id)
  }

  if (useLlm === null) return <></>

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1 className="brand-title">Flavor<span className="brand-accent">Matrix</span></h1>
          <p className="brand-sub">Molecular Gastronomy Explorer</p>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-btn ${view === 'substitution' ? 'active' : ''}`}
            onClick={() => setView('substitution')}
          >
            Substitution Engine
          </button>
          <button
            className={`nav-btn ${view === 'network' ? 'active' : ''}`}
            onClick={() => setView('network')}
          >
            Flavor Universe
          </button>
          <button
            className={`nav-btn ${view === 'sensory' ? 'active' : ''}`}
            onClick={() => setView('sensory')}
          >
            Sensory Map
          </button>
        </nav>

        <FeedbackPanel />
      </aside>

      {/* Main panel */}
      <main className="main-panel">
        <div className="main-content">
          {view === 'substitution' && (
            <SubstitutionEngine onSelectIngredient={handleSelectIngredient} />
          )}
          {view === 'network' && (
            <FlavorNetwork
              selectedIngredient={selectedIngredient}
              onSelectIngredient={handleSelectIngredient}
            />
          )}
          {view === 'sensory' && (
            <SensoryMap3D onSelectIngredient={handleSelectIngredient} />
          )}
        </div>

        {/* Ingredient profile side panel */}
        <IngredientProfileTable
          ingredientId={profileId}
          onClose={() => setProfileId(null)}
        />
      </main>

      {/* Chat bubble */}
      {useLlm && (
        <>
          <button
            className={`chat-toggle ${chatOpen ? 'open' : ''}`}
            onClick={() => setChatOpen(!chatOpen)}
            title="AI Flavor Chemist"
          >
            {chatOpen ? '✕' : '💬'}
          </button>
          {chatOpen && <Chat />}
        </>
      )}
    </div>
  )
}

export default App
