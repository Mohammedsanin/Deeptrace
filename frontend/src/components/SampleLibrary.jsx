import { useState } from 'react'
import { motion } from 'framer-motion'
import './SampleLibrary.css'

const SAMPLE_FILES = [
    {
        id: 1,
        name: 'Real Audio Sample 1',
        type: 'audio',
        category: 'real',
        description: 'Authentic human voice recording',
        icon: '🎤',
        filename: 'LA_T_1000137.mp3',
        path: '/test_files/audio/LA_T_1000137.mp3'
    },
    {
        id: 2,
        name: 'Real Audio Sample 2',
        type: 'audio',
        category: 'real',
        description: 'Genuine interview recording',
        icon: '🎙️',
        filename: 'LA_T_1000406.mp3',
        path: '/test_files/audio/LA_T_1000406.mp3'
    },
    {
        id: 3,
        name: 'Real Audio Sample 3',
        type: 'audio',
        category: 'real',
        description: 'Natural speech sample',
        icon: '🗣️',
        filename: 'LA_T_1000648.mp3',
        path: '/test_files/audio/LA_T_1000648.mp3'
    },
    {
        id: 4,
        name: 'Real Video Clip 1',
        type: 'video',
        category: 'real',
        description: 'Authentic video footage',
        icon: '🎬',
        filename: 'real_1.mp4',
        path: '/test_files/video/real_1.mp4'
    },
    {
        id: 5,
        name: 'Real Video Clip 2',
        type: 'video',
        category: 'real',
        description: 'Genuine video recording',
        icon: '📹',
        filename: 'real_2.mp4',
        path: '/test_files/video/real_2.mp4'
    },
    {
        id: 6,
        name: 'Deepfake Video 1',
        type: 'video',
        category: 'fake',
        description: 'AI-manipulated video content',
        icon: '⚠️',
        filename: 'fake_1.mp4',
        path: '/test_files/video/fake_1.mp4'
    },
    {
        id: 7,
        name: 'Deepfake Video 2',
        type: 'video',
        category: 'fake',
        description: 'Synthetic face-swapped video',
        icon: '🎭',
        filename: 'fake_2.mp4',
        path: '/test_files/video/fake_2.mp4'
    },
    {
        id: 8,
        name: 'Deepfake Video 3',
        type: 'video',
        category: 'fake',
        description: 'AI-generated deepfake',
        icon: '🤖',
        filename: 'fake_3.mp4',
        path: '/test_files/video/fake_3.mp4'
    }
]

function SampleLibrary({ onFileSelect, onAnalyze, selectedFile, isAnalyzing }) {
    const [selectedSample, setSelectedSample] = useState(null)
    const [filter, setFilter] = useState('all') // all, audio, video
    const [loading, setLoading] = useState(false)
    const [previewUrl, setPreviewUrl] = useState(null)
    const [previewType, setPreviewType] = useState(null)

    const handleSampleClick = async (sample) => {
        setSelectedSample(sample.id)
        setLoading(true)

        try {
            // Fetch the actual file from the backend (port 8000)
            const response = await fetch(`http://localhost:8000${sample.path}`, {
                mode: 'cors'
            })

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`)
            }

            // Get the blob
            const blob = await response.blob()

            // Create preview URL
            const url = URL.createObjectURL(blob)
            setPreviewUrl(url)
            setPreviewType(sample.type)

            // Create a File object from the blob
            const file = new File([blob], sample.filename, {
                type: sample.type === 'audio' ? 'audio/mpeg' : 'video/mp4',
                lastModified: Date.now()
            })

            // Pass the file to the parent component (but don't auto-analyze)
            onFileSelect(file)

        } catch (error) {
            console.error('Error loading sample file:', error)
            alert(`Failed to load sample file: ${error.message}\n\nMake sure the backend is running on port 8000.`)
        } finally {
            setLoading(false)
        }
    }

    const handleAnalyzeClick = () => {
        if (onAnalyze) {
            onAnalyze()
        }
    }

    const filteredSamples = SAMPLE_FILES.filter(sample => {
        if (filter === 'all') return true
        return sample.type === filter
    })

    return (
        <div className="sample-library glass-card fade-in">
            <div className="library-header">
                <h2 className="section-title">
                    <span className="title-icon">🎯</span>
                    Try Sample Files
                </h2>
                <p className="section-description">
                    Test the detection system with real examples from our dataset
                </p>
            </div>

            <div className="filter-tabs">
                <button
                    className={`filter-tab ${filter === 'all' ? 'active' : ''}`}
                    onClick={() => setFilter('all')}
                >
                    All Samples
                </button>
                <button
                    className={`filter-tab ${filter === 'audio' ? 'active' : ''}`}
                    onClick={() => setFilter('audio')}
                >
                    🎵 Audio
                </button>
                <button
                    className={`filter-tab ${filter === 'video' ? 'active' : ''}`}
                    onClick={() => setFilter('video')}
                >
                    🎬 Video
                </button>
            </div>

            {previewUrl && (
                <motion.div
                    className="preview-section glass-card"
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                >
                    <div className="preview-header">
                        <h4>📺 Preview</h4>
                        <button
                            className="btn btn-primary analyze-preview-btn"
                            onClick={handleAnalyzeClick}
                            disabled={isAnalyzing}
                        >
                            {isAnalyzing ? (
                                <>
                                    <span className="spinner"></span>
                                    <span>Analyzing...</span>
                                </>
                            ) : (
                                <>
                                    <span>🔍</span>
                                    <span>Analyze This Sample</span>
                                </>
                            )}
                        </button>
                    </div>
                    {previewType === 'audio' ? (
                        <audio controls src={previewUrl} className="audio-preview">
                            Your browser does not support the audio element.
                        </audio>
                    ) : (
                        <video controls src={previewUrl} className="video-preview">
                            Your browser does not support the video element.
                        </video>
                    )}
                </motion.div>
            )}

            <div className="samples-grid">
                {filteredSamples.map((sample, index) => (
                    <motion.div
                        key={sample.id}
                        className={`sample-card ${selectedSample === sample.id ? 'selected' : ''} ${sample.category}`}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        onClick={() => !loading && !isAnalyzing && handleSampleClick(sample)}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                    >
                        <div className="sample-icon">{sample.icon}</div>
                        <div className="sample-info">
                            <h4 className="sample-name">{sample.name}</h4>
                            <p className="sample-description">{sample.description}</p>
                        </div>
                        <div className={`sample-badge ${sample.category}`}>
                            {sample.category === 'real' ? '✓ Real' : '⚠ Fake'}
                        </div>
                    </motion.div>
                ))}
            </div>

            {(loading || isAnalyzing) && (
                <div className="analyzing-overlay">
                    <div className="spinner"></div>
                    <p>{loading ? 'Loading sample...' : 'Analyzing...'}</p>
                </div>
            )}
        </div>
    )
}

export default SampleLibrary
