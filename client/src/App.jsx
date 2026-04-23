import { useEffect, useRef, useState } from 'react'
import './App.css'

const POLL_INTERVAL_MS = 4000

const statusTone = {
  indexed: 'success',
  indexing: 'warning',
  uploaded: 'warning',
  error: 'danger',
}

const formatStatus = (status) => {
  if (!status) {
    return 'Unknown'
  }

  return status.charAt(0).toUpperCase() + status.slice(1)
}

const formatTime = (value) => {
  if (!value) {
    return 'Just now'
  }

  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
    }).format(new Date(value))
  } catch {
    return 'Just now'
  }
}

const shortSnippet = (snippet) => {
  if (!snippet) {
    return ''
  }

  return snippet.length > 420 ? `${snippet.slice(0, 420)}\n...` : snippet
}

function App() {
  const [repositories, setRepositories] = useState([])
  const [selectedRepoId, setSelectedRepoId] = useState(null)
  const [messages, setMessages] = useState([])
  const [query, setQuery] = useState('')
  const [isLandingVisible, setIsLandingVisible] = useState(true)
  const [isLoadingRepos, setIsLoadingRepos] = useState(true)
  const [isUploading, setIsUploading] = useState(false)
  const [isQuerying, setIsQuerying] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [notice, setNotice] = useState('')
  const fileInputRef = useRef(null)
  const workspaceRef = useRef(null)

  const selectedRepo =
    repositories.find((repo) => repo.id === selectedRepoId) ?? null

  useEffect(() => {
    let cancelled = false

    const loadRepositories = async ({ keepSelection = true } = {}) => {
      try {
        if (!keepSelection) {
          setIsLoadingRepos(true)
        }

        const response = await fetch('/api/repositories')
        if (!response.ok) {
          throw new Error('Unable to load repositories.')
        }

        const data = await response.json()
        if (cancelled) {
          return
        }

        setRepositories(data.repositories)
        setSelectedRepoId((current) => {
          if (current && data.repositories.some((repo) => repo.id === current)) {
            return current
          }

          return data.repositories[0]?.id ?? null
        })
      } catch (error) {
        if (!cancelled) {
          setNotice(error.message)
        }
      } finally {
        if (!cancelled) {
          setIsLoadingRepos(false)
        }
      }
    }

    loadRepositories({ keepSelection: false })

    const interval = window.setInterval(() => {
      loadRepositories()
    }, POLL_INTERVAL_MS)

    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [])

  useEffect(() => {
    setMessages([])
  }, [selectedRepoId])

  const openWorkspace = () => {
    setIsLandingVisible(false)
    window.setTimeout(() => {
      workspaceRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 60)
  }

  const refreshRepositories = async () => {
    const response = await fetch('/api/repositories')
    if (!response.ok) {
      throw new Error('Unable to refresh repositories.')
    }

    const data = await response.json()
    setRepositories(data.repositories)
    setSelectedRepoId((current) => {
      if (current && data.repositories.some((repo) => repo.id === current)) {
        return current
      }

      return data.repositories[0]?.id ?? null
    })
  }

  const handleFileUpload = async (file) => {
    if (!file || !file.name.toLowerCase().endsWith('.zip')) {
      setNotice('Please upload a ZIP archive.')
      return
    }

    setIsUploading(true)
    setNotice('')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('/api/repositories/upload', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail ?? 'Upload failed.')
      }

      await refreshRepositories()
      setSelectedRepoId(data.repository.id)
      setMessages([])
      setIsLandingVisible(false)
      setNotice(`Repository "${data.repository.name}" uploaded and indexing started.`)
    } catch (error) {
      setNotice(error.message)
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const onDrop = (event) => {
    event.preventDefault()
    setDragActive(false)
    handleFileUpload(event.dataTransfer.files?.[0])
  }

  const submitQuery = async (event) => {
    event.preventDefault()

    if (!selectedRepo || !query.trim()) {
      return
    }

    const question = query.trim()

    setMessages((current) => [
      ...current,
      {
        role: 'user',
        content: question,
      },
    ])
    setQuery('')
    setIsQuerying(true)
    setNotice('')

    try {
      const response = await fetch(`/api/repositories/${selectedRepo.id}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: question }),
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail ?? 'Query failed.')
      }

      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          content: data.answer,
          references: data.references,
          generatedWithModel: data.generated_with_model,
        },
      ])
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          content: error.message,
          references: [],
          generatedWithModel: false,
        },
      ])
    } finally {
      setIsQuerying(false)
    }
  }

  return (
    <div className="page-shell">
      {isLandingVisible ? (
        <main className="landing-shell">
          <section className="hero-panel">
            <div className="hero-copy">
              <span className="eyebrow">AI codebase intelligence</span>
              <h1>Understand Your Codebase Instantly</h1>
              <p>
                Upload a repository, index its architecture, and ask natural-language
                questions across functions, dependencies, and embeddings in one calm
                workspace.
              </p>
              <div className="hero-actions">
                <button type="button" className="primary-button" onClick={openWorkspace}>
                  Get Started
                </button>
                <a href="#how-it-works" className="secondary-link">
                  See how it works
                </a>
              </div>
            </div>

            <div className="hero-visual" aria-hidden="true">
              <div className="visual-card">
                <div className="visual-header">
                  <span className="visual-dot amber"></span>
                  <span className="visual-dot sage"></span>
                  <span className="visual-dot slate"></span>
                </div>
                <div className="visual-code">
                  <span>repo.scan()</span>
                  <span>functions.extract()</span>
                  <span>graph.resolve()</span>
                  <span>assistant.answer()</span>
                </div>
                <div className="visual-map">
                  <div className="map-node large"></div>
                  <div className="map-node medium"></div>
                  <div className="map-node medium"></div>
                  <div className="map-node small"></div>
                </div>
              </div>
            </div>
          </section>

          <section id="how-it-works" className="info-section">
            <div className="section-heading">
              <span className="eyebrow">How it works</span>
              <h2>From archive to answers in three steps</h2>
            </div>
            <div className="steps-grid">
              <article className="info-card">
                <span className="card-index">01</span>
                <h3>Upload</h3>
                <p>Drop in a ZIP archive and create a clean repository workspace.</p>
              </article>
              <article className="info-card">
                <span className="card-index">02</span>
                <h3>Index</h3>
                <p>Extract functions, build dependency graphs, and generate embeddings.</p>
              </article>
              <article className="info-card">
                <span className="card-index">03</span>
                <h3>Query</h3>
                <p>Ask questions in plain English and inspect the referenced code directly.</p>
              </article>
            </div>
          </section>

          <section className="info-section feature-grid">
            <article className="feature-card">
              <h3>Function Extraction</h3>
              <p>Surface implementation-level context with precise file and line references.</p>
            </article>
            <article className="feature-card">
              <h3>Dependency Graph</h3>
              <p>Trace relationships between components before you open a single file.</p>
            </article>
            <article className="feature-card">
              <h3>Embeddings Search</h3>
              <p>Blend semantic relevance with code-aware retrieval for faster answers.</p>
            </article>
          </section>

          <footer className="landing-footer">
            <span>RepoRAG</span>
            <span>Built for focused developer workflows</span>
          </footer>
        </main>
      ) : null}

      <main className="workspace-shell" ref={workspaceRef}>
        <aside className="sidebar">
          <div className="sidebar-header">
            <div>
              <span className="eyebrow">Workspace</span>
              <h2>Repositories</h2>
            </div>
            <p>Indexed projects update automatically from the backend storage.</p>
          </div>

          <div className="repo-list">
            {isLoadingRepos ? (
              <div className="repo-empty">Loading repositories...</div>
            ) : repositories.length === 0 ? (
              <div className="repo-empty">No repositories indexed yet.</div>
            ) : (
              repositories.map((repo) => (
                <button
                  key={repo.id}
                  type="button"
                  className={`repo-item ${repo.id === selectedRepoId ? 'active' : ''}`}
                  onClick={() => setSelectedRepoId(repo.id)}
                >
                  <div className="repo-item-top">
                    <span className="repo-name">{repo.name}</span>
                    <span className={`status-pill ${statusTone[repo.status] ?? ''}`}>
                      {formatStatus(repo.status)}
                    </span>
                  </div>
                  <div className="repo-meta">
                    <span>{repo.function_count} functions</span>
                    <span>{formatTime(repo.updated_at)}</span>
                  </div>
                </button>
              ))
            )}
          </div>

          <button
            type="button"
            className="upload-sidebar-button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
          >
            + Upload New Repository
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".zip"
            hidden
            onChange={(event) => handleFileUpload(event.target.files?.[0])}
          />
        </aside>

        <section className="main-panel">
          <header className="main-panel-header">
            <div>
              <span className="eyebrow">Assistant</span>
              <h2>{selectedRepo?.name ?? 'Repository Workspace'}</h2>
            </div>
            {selectedRepo ? (
              <div className="header-meta">
                <span className={`status-pill ${statusTone[selectedRepo.status] ?? ''}`}>
                  {formatStatus(selectedRepo.status)}
                </span>
                <span>{selectedRepo.embedding_ready ? 'Embeddings ready' : 'Preparing embeddings'}</span>
              </div>
            ) : null}
          </header>

          {notice ? <div className="notice-banner">{notice}</div> : null}

          {!selectedRepo ? (
            <div
              className={`empty-state upload-state ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={(event) => {
                event.preventDefault()
                setDragActive(true)
              }}
              onDragOver={(event) => {
                event.preventDefault()
                setDragActive(true)
              }}
              onDragLeave={(event) => {
                event.preventDefault()
                setDragActive(false)
              }}
              onDrop={onDrop}
            >
              <div className="upload-icon">ZIP</div>
              <h3>Upload a repository to begin</h3>
              <p>Drag and drop a ZIP archive or choose a repository to index.</p>
              <button
                type="button"
                className="primary-button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
              >
                {isUploading ? 'Uploading...' : 'Upload ZIP'}
              </button>
            </div>
          ) : selectedRepo.status === 'indexing' || selectedRepo.status === 'uploaded' ? (
            <div className="empty-state indexing-state">
              <div className="spinner-ring"></div>
              <h3>Indexing repository...</h3>
              <p>Extracting functions and building embeddings for {selectedRepo.name}.</p>
              <div className="progress-track">
                <div className="progress-bar"></div>
              </div>
            </div>
          ) : selectedRepo.status === 'error' ? (
            <div className="empty-state error-state">
              <h3>Indexing failed</h3>
              <p>{selectedRepo.error_message ?? 'Something went wrong while processing this repository.'}</p>
            </div>
          ) : (
            <div className="chat-shell">
              <div className="chat-stream">
                {messages.length === 0 ? (
                  <div className="chat-placeholder">
                    <h3>Ask questions about your codebase</h3>
                    <p>
                      Try questions like "Where is authentication handled?" or
                      "Which function builds the dependency graph?"
                    </p>
                  </div>
                ) : (
                  messages.map((message, index) => (
                    <article
                      key={`${message.role}-${index}`}
                      className={`message-card ${message.role}`}
                    >
                      <div className="message-label">
                        {message.role === 'user' ? 'You' : 'Assistant'}
                      </div>
                      <p className="message-content">{message.content}</p>
                      {message.references?.length ? (
                        <div className="reference-list">
                          {message.references.map((reference) => (
                            <section key={reference.id} className="reference-card">
                              <div className="reference-header">
                                <div>
                                  <strong>{reference.name}</strong>
                                  <span>{reference.file_path}</span>
                                </div>
                                <span>
                                  {reference.start_line}-{reference.end_line}
                                </span>
                              </div>
                              <pre>
                                <code>{shortSnippet(reference.source_code)}</code>
                              </pre>
                            </section>
                          ))}
                        </div>
                      ) : null}
                    </article>
                  ))
                )}
              </div>

              <form className="query-form" onSubmit={submitQuery}>
                <input
                  type="text"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Ask questions about your codebase..."
                  disabled={isQuerying}
                />
                <button type="submit" className="primary-button" disabled={isQuerying}>
                  {isQuerying ? 'Thinking...' : 'Send'}
                </button>
              </form>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
