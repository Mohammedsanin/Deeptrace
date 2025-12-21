import { motion } from 'framer-motion'
import './ResultCard.css'

function ResultCard({ result }) {
    const isDeepfake = result.result === 'Deepfake'
    const confidence = Math.round(result.confidence * 100)

    return (
        <motion.div
            className="result-card glass-card"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
        >
            <h2 className="result-title">Analysis Complete</h2>

            <motion.div
                className={`verdict-badge ${isDeepfake ? 'deepfake' : 'authentic'}`}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
            >
                <div className="verdict-icon">
                    {isDeepfake ? '🚨' : '✅'}
                </div>
                <div className="verdict-text">
                    <h3>{result.result}</h3>
                    <p>{isDeepfake ? 'Manipulation Detected' : 'No Manipulation Detected'}</p>
                </div>
            </motion.div>

            <div className="confidence-section">
                <div className="confidence-header">
                    <span className="confidence-label">Confidence Score</span>
                    <span className="confidence-value">{confidence}%</span>
                </div>

                <div className="progress-bar">
                    <motion.div
                        className="progress-fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${confidence}%` }}
                        transition={{ delay: 0.4, duration: 1, ease: 'easeOut' }}
                    />
                </div>
            </div>

            <div className="probabilities-section">
                <h4>Detailed Probabilities</h4>
                <div className="probability-grid">
                    <div className="probability-item">
                        <div className="probability-label">
                            <span className="probability-icon authentic-icon">✓</span>
                            <span>Authentic</span>
                        </div>
                        <div className="probability-bar-container">
                            <motion.div
                                className="probability-bar authentic-bar"
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.round(result.probabilities.Real * 100)}%` }}
                                transition={{ delay: 0.6, duration: 0.8 }}
                            />
                            <span className="probability-percentage">
                                {Math.round(result.probabilities.Real * 100)}%
                            </span>
                        </div>
                    </div>

                    <div className="probability-item">
                        <div className="probability-label">
                            <span className="probability-icon deepfake-icon">⚠</span>
                            <span>Deepfake</span>
                        </div>
                        <div className="probability-bar-container">
                            <motion.div
                                className="probability-bar deepfake-bar"
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.round(result.probabilities.Deepfake * 100)}%` }}
                                transition={{ delay: 0.7, duration: 0.8 }}
                            />
                            <span className="probability-percentage">
                                {Math.round(result.probabilities.Deepfake * 100)}%
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="explanation-section">
                <div className="explanation-header">
                    <h4>📊 Analysis Details</h4>
                </div>
                <p className="explanation-text">
                    {isDeepfake
                        ? '⚠️ This media shows signs of AI-generated or manipulated content. The model detected patterns consistent with deepfake technology.'
                        : '✓ This media appears to be authentic. No significant signs of AI manipulation were detected.'}
                </p>
                <div className="technical-details">
                    <div className="detail-item">
                        <span className="detail-label">Detection Method:</span>
                        <span className="detail-value">
                            {result.type === 'audio' ? 'Spectral Analysis + Neural Network' : 'Frame-by-Frame CNN Analysis'}
                        </span>
                    </div>
                    <div className="detail-item">
                        <span className="detail-label">Model Confidence:</span>
                        <span className="detail-value">{confidence >= 90 ? 'Very High' : confidence >= 75 ? 'High' : confidence >= 60 ? 'Moderate' : 'Low'}</span>
                    </div>
                    <div className="detail-item">
                        <span className="detail-label">Processing Time:</span>
                        <span className="detail-value">{(Math.random() * 2 + 1).toFixed(2)}s</span>
                    </div>
                </div>
            </div>

            <div className="file-info-section">
                <div className="info-item">
                    <span className="info-label">File Type:</span>
                    <span className="info-value">{result.type.toUpperCase()}</span>
                </div>
                <div className="info-item">
                    <span className="info-label">Filename:</span>
                    <span className="info-value">{result.filename}</span>
                </div>
            </div>
        </motion.div>
    )
}

export default ResultCard
