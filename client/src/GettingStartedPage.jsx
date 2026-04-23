import Footer from './Footer.jsx'

function GettingStartedPage({ navigate }) {
  return (
    <div className="page-shell">
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
              <button
                type="button"
                className="primary-button"
                onClick={() => navigate('/workspace')}
              >
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

      </main>
      <Footer />
    </div>
  )
}

export default GettingStartedPage
