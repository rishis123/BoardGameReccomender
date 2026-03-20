import { useState, useRef, useEffect } from 'react'
import { ChatMessage } from './types'

function Chat(): JSX.Element {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const msgIdRef = useRef(0)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendFeedback = async (msgId: string, score: number) => {
    await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ context_type: 'chat', context_id: msgId, score }),
    })
  }

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    setMessages(prev => [...prev, { text, isUser: true }])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })

      if (!response.ok) {
        const data = await response.json()
        setMessages(prev => [...prev, { text: 'Error: ' + (data.error || response.status), isUser: false }])
        setLoading(false)
        return
      }

      const currentId = String(++msgIdRef.current)
      let assistantText = ''
      let citations: string[] = []
      setMessages(prev => [...prev, { text: '', isUser: false, id: currentId }])
      setLoading(false)

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.citations) citations = data.citations
              if (data.error) {
                setMessages(prev => [
                  ...prev.slice(0, -1),
                  { text: 'Error: ' + data.error, isUser: false, id: currentId },
                ])
                return
              }
              if (data.content !== undefined) {
                assistantText += data.content
                setMessages(prev => [
                  ...prev.slice(0, -1),
                  { text: assistantText, isUser: false, citations, id: currentId },
                ])
              }
            } catch { /* skip malformed lines */ }
          }
        }
      }
    } catch {
      setMessages(prev => [...prev, { text: 'Something went wrong. Check the console.', isUser: false }])
      setLoading(false)
    }
  }

  return (
    <div className="chat-drawer">
      <div className="chat-drawer-header">AI Flavor Chemist</div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg ${msg.isUser ? 'user' : 'assistant'}`}>
            <p>{msg.text}</p>
            {!msg.isUser && msg.citations && msg.citations.length > 0 && (
              <div className="chat-citations">
                {msg.citations.map((c, ci) => (
                  <span key={ci} className="citation-tag">{c}</span>
                ))}
              </div>
            )}
            {!msg.isUser && msg.id && msg.text && (
              <div className="chat-feedback">
                <button onClick={() => sendFeedback(msg.id!, 1)} title="Helpful">+1</button>
                <button onClick={() => sendFeedback(msg.id!, -1)} title="Not helpful">-1</button>
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="loading-indicator visible">
            <span className="loading-dot" />
            <span className="loading-dot" />
            <span className="loading-dot" />
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form className="chat-input-form" onSubmit={sendMessage}>
        <input
          type="text"
          placeholder="Ask about flavor science…"
          value={input}
          onChange={e => setInput(e.target.value)}
          disabled={loading}
          autoComplete="off"
        />
        <button type="submit" disabled={loading}>Send</button>
      </form>
    </div>
  )
}

export default Chat
