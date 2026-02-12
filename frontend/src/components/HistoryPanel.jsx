import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import './HistoryPanel.css'

function HistoryPanel({ onHistoryItemClick }) {
    const [history, setHistory] = useState([])
    const [isExpanded, setIsExpanded] = useState(false)

    useEffect(() => {
        // Load history from localStorage
        const savedHistory = localStorage.getItem('deeptrace_history')
        if (savedHistory) {
            setHistory(JSON.parse(savedHistory))
        }
    }, [])

    const clearHistory = () => {
        if (confirm('Are you sure you want to clear all history?')) {
            localStorage.removeItem('deeptrace_history')
            setHistory([])
        }
    }

    const removeItem = (index, e) => {
        e.stopPropagation()
        const newHistory = history.filter((_, i) => i !== index)
        setHistory(newHistory)
        localStorage.setItem('deeptrace_history', JSON.stringify(newHistory))
    }

    const formatTimestamp = (timestamp) => {
        const date = new Date(timestamp)
        const now = new Date()
        const diffMs = now - date
        const diffMins = Math.floor(diffMs / 60000)

        if (diffMins < 1) return 'Just now'
        if (diffMins < 60) return `${diffMins}m ago`
        if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`
        return date.toLocaleDateString()
    }

    if (history.length === 0) {
        return null
    }

    return (
        <div className={`history-panel glass-card ${isExpanded ? 'expanded' : ''}`}>
            <div className="history-header" onClick={() => setIsExpanded(!isExpanded)}>
                <div className="history-title">
                    <span className="history-icon">📋</span>
                    <h3>Recent Analyses</h3>
                    <span className="history-count">{history.length}</span>
                </div>
                <button className="expand-btn">
                    {isExpanded ? '▼' : '▲'}
                </button>
            </div>

            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        className="history-content"
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                    >
                        <div className="history-list">
                            {history.map((item, index) => (
                                <motion.div
                                    key={index}
                                    className={`history-item ${item.result.result.toLowerCase()}`}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: index * 0.05 }}
                                    onClick={() => onHistoryItemClick && onHistoryItemClick(item)}
                                >
                                    <div className="history-item-icon">
                                        {item.result.type === 'audio' ? '🎵' : '🎬'}
                                    </div>
                                    <div className="history-item-info">
                                        <div className="history-item-name">{item.filename}</div>
                                        <div className="history-item-meta">
                                            <span className={`history-result ${item.result.result.toLowerCase()}`}>
                                                {item.result.result === 'Real' ? '✓' : '⚠'} {item.result.result}
                                            </span>
                                            <span className="history-confidence">
                                                {Math.round(item.result.confidence * 100)}%
                                            </span>
                                            <span className="history-time">
                                                {formatTimestamp(item.timestamp)}
                                            </span>
                                        </div>
                                    </div>
                                    <button
                                        className="remove-btn"
                                        onClick={(e) => removeItem(index, e)}
                                        title="Remove from history"
                                    >
                                        ×
                                    </button>
                                </motion.div>
                            ))}
                        </div>

                        <button className="clear-history-btn" onClick={clearHistory}>
                            🗑️ Clear All History
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

// Helper function to add items to history (export this)
export const addToHistory = (filename, result) => {
    const historyItem = {
        filename,
        result,
        timestamp: Date.now()
    }

    const savedHistory = localStorage.getItem('deeptrace_history')
    let history = savedHistory ? JSON.parse(savedHistory) : []

    // Add to beginning and limit to 10 items
    history.unshift(historyItem)
    history = history.slice(0, 10)

    localStorage.setItem('deeptrace_history', JSON.stringify(history))
}

export default HistoryPanel
