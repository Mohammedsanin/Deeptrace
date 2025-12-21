import { useState, useRef } from 'react'
import './UploadSection.css'

function UploadSection({ onFileSelect, onAnalyze, selectedFile, isAnalyzing }) {
    const [isDragging, setIsDragging] = useState(false)
    const fileInputRef = useRef(null)

    const handleDragOver = (e) => {
        e.preventDefault()
        setIsDragging(true)
    }

    const handleDragLeave = () => {
        setIsDragging(false)
    }

    const handleDrop = (e) => {
        e.preventDefault()
        setIsDragging(false)

        const files = e.dataTransfer.files
        if (files.length > 0) {
            handleFileChange(files[0])
        }
    }

    const handleFileChange = (file) => {
        // Validate file type
        const isAudio = file.type.startsWith('audio/')
        const isVideo = file.type.startsWith('video/')

        if (!isAudio && !isVideo) {
            alert('Please select an audio or video file')
            return
        }

        onFileSelect(file)
    }

    const handleFileInputChange = (e) => {
        if (e.target.files.length > 0) {
            handleFileChange(e.target.files[0])
        }
    }

    const handleBrowseClick = () => {
        fileInputRef.current?.click()
    }

    const getFileIcon = () => {
        if (!selectedFile) return '📁'
        if (selectedFile.type.startsWith('audio/')) return '🎵'
        if (selectedFile.type.startsWith('video/')) return '🎬'
        return '📄'
    }

    const formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes'
        const k = 1024
        const sizes = ['Bytes', 'KB', 'MB', 'GB']
        const i = Math.floor(Math.log(bytes) / Math.log(k))
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
    }

    return (
        <div className="upload-section glass-card fade-in">
            <h2 className="section-title">
                Upload Media for Analysis
            </h2>
            <p className="section-description">
                Upload an audio or video file to detect deepfake manipulation
            </p>

            <div
                className={`drop-zone ${isDragging ? 'dragging' : ''} ${selectedFile ? 'has-file' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={handleBrowseClick}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*,video/*"
                    onChange={handleFileInputChange}
                    style={{ display: 'none' }}
                />

                <div className="drop-zone-content">
                    <div className="file-icon">{getFileIcon()}</div>

                    {selectedFile ? (
                        <div className="file-info">
                            <h3 className="file-name">{selectedFile.name}</h3>
                            <p className="file-details">
                                {selectedFile.type.split('/')[0].toUpperCase()} • {formatFileSize(selectedFile.size)}
                            </p>
                        </div>
                    ) : (
                        <div className="upload-prompt">
                            <h3>Drag & Drop or Click to Browse</h3>
                            <p>Supports: MP3, WAV, MP4, AVI, MOV (Max 100MB)</p>
                        </div>
                    )}
                </div>
            </div>

            {selectedFile && (
                <button
                    className="btn btn-primary analyze-btn"
                    onClick={onAnalyze}
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
                            <span>Analyze for Deepfake</span>
                        </>
                    )}
                </button>
            )}
        </div>
    )
}

export default UploadSection
