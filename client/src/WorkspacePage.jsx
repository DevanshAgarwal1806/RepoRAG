import { useEffect, useRef, useState } from 'react'
import Footer from './Footer.jsx'

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

const messagesFromHistory = (history = []) =>
  history.flatMap((entry, index) => {
    const id = entry.id ?? `history-${index}`

    return [
      {
        id: `${id}-user`,
        role: 'user',
        content: entry.query,
        createdAt: entry.created_at,
      },
      {
        id: `${id}-assistant`,
        role: 'assistant',
        content: entry.answer,
        references: entry.references ?? [],
        generatedWithModel: entry.generated_with_model,
        createdAt: entry.created_at,
      },
    ]
  })

const renderInlineMarkdown = (text) => {
  const source = text ?? ''
  const inlinePattern = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g
  const parts = []
  let lastIndex = 0
  let match

  while ((match = inlinePattern.exec(source)) !== null) {
    if (match.index > lastIndex) {
      parts.push(source.slice(lastIndex, match.index))
    }

    parts.push(match[0])
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < source.length) {
    parts.push(source.slice(lastIndex))
  }

  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index}>{part.slice(2, -2)}</strong>
    }

    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code key={index} className="message-inline-code">
          {part.slice(1, -1)}
        </code>
      )
    }

    const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/)
    if (linkMatch) {
      const [, label, href] = linkMatch
      return (
        <a
          key={index}
          href={href}
          className="message-link"
          target="_blank"
          rel="noreferrer"
        >
          {label}
        </a>
      )
    }

    return part
  })
}

const renderMarkdownText = (text, keyPrefix) => {
  const lines = text.split('\n')
  const nodes = []
  let paragraphLines = []
  let listItems = []

  const flushParagraph = () => {
    if (paragraphLines.length === 0) {
      return
    }

    const paragraph = paragraphLines.join(' ').trim()
    if (paragraph) {
      nodes.push(
        <p key={`${keyPrefix}-p-${nodes.length}`} className="message-content">
          {renderInlineMarkdown(paragraph)}
        </p>,
      )
    }
    paragraphLines = []
  }

  const flushList = () => {
    if (listItems.length === 0) {
      return
    }

    nodes.push(
      <ul key={`${keyPrefix}-ul-${nodes.length}`} className="message-list">
        {listItems.map((item, index) => (
          <li key={`${keyPrefix}-li-${index}`}>{renderInlineMarkdown(item)}</li>
        ))}
      </ul>,
    )
    listItems = []
  }

  lines.forEach((line) => {
    const trimmed = line.trim()

    if (!trimmed) {
      flushParagraph()
      flushList()
      return
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/)
    if (headingMatch) {
      flushParagraph()
      flushList()

      const level = headingMatch[1].length
      const headingText = headingMatch[2].trim()

      nodes.push(
        <div
          key={`${keyPrefix}-h-${nodes.length}`}
          className={`message-heading heading-${level}`}
        >
          {renderInlineMarkdown(headingText)}
        </div>,
      )
      return
    }

    const listMatch = trimmed.match(/^[-*]\s+(.*)$/)
    if (listMatch) {
      flushParagraph()
      listItems.push(listMatch[1].trim())
      return
    }

    flushList()
    paragraphLines.push(trimmed)
  })

  flushParagraph()
  flushList()

  return nodes
}

const renderMessageContent = (content) => {
  const normalizedContent = content ?? ''
  const fencePattern = /```([\w-]+)?\n([\s\S]*?)```/g
  const nodes = []
  let lastIndex = 0
  let match

  while ((match = fencePattern.exec(normalizedContent)) !== null) {
    const [fullMatch, language = '', code] = match

    if (match.index > lastIndex) {
      nodes.push(
        ...renderMarkdownText(
          normalizedContent.slice(lastIndex, match.index),
          `text-${nodes.length}`,
        ),
      )
    }

    nodes.push(
      <div key={`code-${nodes.length}`} className="message-code-block">
        {language ? (
          <div className="message-code-label">{language}</div>
        ) : null}
        <pre>
          <code>{code.trimEnd()}</code>
        </pre>
      </div>,
    )

    lastIndex = match.index + fullMatch.length
  }

  if (lastIndex < normalizedContent.length) {
    nodes.push(
      ...renderMarkdownText(
        normalizedContent.slice(lastIndex),
        `text-${nodes.length}`,
      ),
    )
  }

  if (nodes.length === 0) {
    return renderMarkdownText(normalizedContent, 'text-empty')
  }

  return nodes
}

function WorkspacePage({ navigate }) {
  const [repositories, setRepositories] = useState([])
  const [selectedRepoId, setSelectedRepoId] = useState(null)
  const [messages, setMessages] = useState([])
  const [query, setQuery] = useState('')
  const [isLoadingRepos, setIsLoadingRepos] = useState(true)
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isQuerying, setIsQuerying] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [notice, setNotice] = useState('')
  const chatStreamRef = useRef(null)
  const fileInputRef = useRef(null)

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
    let cancelled = false

    if (!selectedRepoId) {
      setMessages([])
      setIsLoadingHistory(false)
      return () => {
        cancelled = true
      }
    }

    const loadChatHistory = async () => {
      setMessages([])
      setIsLoadingHistory(true)

      try {
        const response = await fetch(
          `/api/repositories/${selectedRepoId}/chat-history`,
        )
        const data = await response.json()
        if (!response.ok) {
          throw new Error(data.detail ?? 'Unable to load chat history.')
        }

        if (!cancelled) {
          setMessages(messagesFromHistory(data.history ?? []))
        }
      } catch (error) {
        if (!cancelled) {
          setMessages([])
          setNotice(error.message)
        }
      } finally {
        if (!cancelled) {
          setIsLoadingHistory(false)
        }
      }
    }

    loadChatHistory()

    return () => {
      cancelled = true
    }
  }, [selectedRepoId])

  useEffect(() => {
    if (!chatStreamRef.current) {
      return
    }

    chatStreamRef.current.scrollTo({
      top: chatStreamRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }, [messages, isLoadingHistory])

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

    if (!selectedRepo || isLoadingHistory || !query.trim()) {
      return
    }

    const question = query.trim()

    setMessages((current) => [
      ...current,
      {
        id: `pending-${Date.now()}-user`,
        role: 'user',
        content: question,
        createdAt: new Date().toISOString(),
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
          id: `${data.history_entry?.id ?? `pending-${Date.now()}`}-assistant`,
          role: 'assistant',
          content: data.answer,
          references: data.references,
          generatedWithModel: data.generated_with_model,
          createdAt: data.history_entry?.created_at ?? new Date().toISOString(),
        },
      ])
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: error.message,
          references: [],
          generatedWithModel: false,
          createdAt: new Date().toISOString(),
        },
      ])
    } finally {
      setIsQuerying(false)
    }
  }

  return (
    <div className="page-shell">
      <main className="workspace-shell">
        <aside className="sidebar">
          <div className="sidebar-header">
            <div>
              <span className="eyebrow">Workspace</span>
              <h2>Repositories</h2>
            </div>
            <p>Indexed projects update automatically from the backend storage.</p>
          </div>

          <button
            type="button"
            className="secondary-route-button"
            onClick={() => navigate('/')}
          >
            Back to Getting Started
          </button>

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
              <div className="chat-stream" ref={chatStreamRef}>
                {isLoadingHistory ? (
                  <div className="chat-placeholder">
                    <h3>Loading saved chat history...</h3>
                    <p>Restoring the previous conversation for this repository.</p>
                  </div>
                ) : messages.length === 0 ? (
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
                      key={message.id ?? `${message.role}-${index}`}
                      className={`message-card ${message.role}`}
                    >
                      <div className="message-label">
                        <span>{message.role === 'user' ? 'You' : 'Assistant'}</span>
                        {message.createdAt ? (
                          <time dateTime={message.createdAt}>{formatTime(message.createdAt)}</time>
                        ) : null}
                      </div>
                      <div className="message-body">{renderMessageContent(message.content)}</div>
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
                  disabled={isQuerying || isLoadingHistory}
                />
                <button
                  type="submit"
                  className="primary-button"
                  disabled={isQuerying || isLoadingHistory}
                >
                  {isQuerying ? 'Thinking...' : 'Send'}
                </button>
              </form>
            </div>
          )}
        </section>
      </main>
      <Footer />
    </div>
  )
}

export default WorkspacePage
