import { useState } from 'react'
import Header from './components/Header'
import UploadSection from './components/UploadSection'
import ResultCard from './components/ResultCard'
import SampleLibrary from './components/SampleLibrary'
import HistoryPanel, { addToHistory } from './components/HistoryPanel'
import './App.css'

function App() {
    const [selectedFile, setSelectedFile] = useState(null)
    const [isAnalyzing, setIsAnalyzing] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [refreshHistory, setRefreshHistory] = useState(0)

    const handleFileSelect = (file) => {
        setSelectedFile(file)
        setResult(null)
        setError(null)
    }

    const handleAnalyze = async () => {
        if (!selectedFile) return

        setIsAnalyzing(true)
        setError(null)
        setResult(null)

        try {
            const formData = new FormData()
            formData.append('file', selectedFile)

            // Determine endpoint based on file type
            const isVideo = selectedFile.type.startsWith('video/')
            const endpoint = isVideo ? '/predict/video' : '/predict/audio'

            const response = await fetch(`http://localhost:8000${endpoint}`, {
                method: 'POST',
                body: formData,
            })

            if (!response.ok) {
                const errorData = await response.json()
                throw new Error(errorData.detail || 'Analysis failed')
            }

            const data = await response.json()
            setResult(data)

            // Add to history
            addToHistory(selectedFile.name, data)
            setRefreshHistory(prev => prev + 1)
        } catch (err) {
            setError(err.message)
        } finally {
            setIsAnalyzing(false)
        }
    }

    const handleHistoryItemClick = (historyItem) => {
        // Display the historical result
        setResult(historyItem.result)
        setSelectedFile({ name: historyItem.filename })
    }

    return (
        <div className="app">
            <Header />

            <main className="main-content">
                <div className="container">
                    <div className="content-wrapper">
                        <SampleLibrary
                            onFileSelect={handleFileSelect}
                            onAnalyze={handleAnalyze}
                            isAnalyzing={isAnalyzing}
                        />

                        <UploadSection
                            onFileSelect={handleFileSelect}
                            onAnalyze={handleAnalyze}
                            selectedFile={selectedFile}
                            isAnalyzing={isAnalyzing}
                        />

                        {error && (
                            <div className="error-message glass-card fade-in">
                                <div className="error-icon">⚠️</div>
                                <div className="error-text">
                                    <h3>Analysis Failed</h3>
                                    <p>{error}</p>
                                </div>
                            </div>
                        )}

                        {result && (
                            <ResultCard result={result} />
                        )}
                    </div>
                </div>
            </main>

            <HistoryPanel
                key={refreshHistory}
                onHistoryItemClick={handleHistoryItemClick}
            />

            <footer className="footer">
                <div className="container">
                    <p>Deeptrace © 2024 - Advanced Deepfake Detection System</p>
                </div>
            </footer>
        </div>
    )
}

export default App
