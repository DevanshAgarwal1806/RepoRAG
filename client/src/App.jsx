import { useEffect, useState } from 'react'
import './App.css'
import GettingStartedPage from './GettingStartedPage.jsx'
import WorkspacePage from './WorkspacePage.jsx'

const routes = {
  '/': GettingStartedPage,
  '/workspace': WorkspacePage,
}

const normalizePath = (path) => {
  if (!path) {
    return '/'
  }

  if (path.length > 1 && path.endsWith('/')) {
    return path.slice(0, -1)
  }

  return path
}

function App() {
  const [path, setPath] = useState(() => normalizePath(window.location.pathname))

  useEffect(() => {
    const syncPath = () => {
      setPath(normalizePath(window.location.pathname))
    }

    window.addEventListener('popstate', syncPath)

    return () => {
      window.removeEventListener('popstate', syncPath)
    }
  }, [])

  const navigate = (nextPath) => {
    const normalizedPath = normalizePath(nextPath)

    if (normalizedPath === path) {
      return
    }

    window.history.pushState({}, '', normalizedPath)
    setPath(normalizedPath)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const ActiveRoute = routes[path] ?? WorkspacePage

  return <ActiveRoute navigate={navigate} />
}

export default App
