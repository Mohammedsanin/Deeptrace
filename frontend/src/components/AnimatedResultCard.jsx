import React from 'react';
import { motion } from 'framer-motion';
import './AnimatedResultCard.css';

const AnimatedResultCard = ({ result }) => {
  if (!result) return null;

  const isReal = result.prediction === 'real';
  const confidence = Math.round(result.confidence * 100);

  return (
    <motion.div
      className="animated-result-card vibrant-card"
      initial={{ opacity: 0, scale: 0.9, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.5, type: "spring" }}
    >
      {/* Verdict Badge with Animation */}
      <motion.div
        className={`verdict-badge-animated ${isReal ? 'real' : 'fake'}`}
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
      >
        <motion.span
          className="verdict-icon"
          animate={{ rotate: [0, 10, -10, 0] }}
          transition={{ delay: 0.4, duration: 0.5 }}
        >
          {isReal ? '✓' : '⚠'}
        </motion.span>
        <span className="verdict-text">
          {isReal ? 'AUTHENTIC' : 'DEEPFAKE DETECTED'}
        </span>
      </motion.div>

      {/* Confidence Meter with Liquid Fill */}
      <div className="confidence-section">
        <div className="confidence-header">
          <span className="confidence-label">Confidence Score</span>
          <motion.span
            className="confidence-value gradient-text"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            {confidence}%
          </motion.span>
        </div>

        <div className="confidence-bar-container">
          <motion.div
            className={`confidence-bar-fill ${isReal ? 'real-fill' : 'fake-fill'}`}
            initial={{ width: 0 }}
            animate={{ width: `${confidence}%` }}
            transition={{ delay: 0.4, duration: 1, ease: "easeOut" }}
          >
            <div className="shimmer-overlay" />
          </motion.div>
        </div>
      </div>

      {/* Metrics Grid with Stagger Animation */}
      {result.features && (
        <motion.div
          className="metrics-grid-animated"
          initial="hidden"
          animate="visible"
          variants={{
            hidden: { opacity: 0 },
            visible: {
              opacity: 1,
              transition: {
                staggerChildren: 0.1,
                delayChildren: 0.5
              }
            }
          }}
        >
          {Object.entries(result.features).map(([key, value], index) => (
            <motion.div
              key={key}
              className="metric-card-animated glass-light"
              variants={{
                hidden: { opacity: 0, y: 20 },
                visible: { opacity: 1, y: 0 }
              }}
            >
              <div className="metric-icon">
                {index === 0 ? '🎯' : index === 1 ? '📊' : '🔍'}
              </div>
              <div className="metric-value-animated">
                {typeof value === 'number' ? value.toFixed(2) : value}
              </div>
              <div className="metric-label-animated">
                {key.replace(/_/g, ' ').toUpperCase()}
              </div>
            </motion.div>
          ))}
        </motion.div>
      )}

      {/* Processing Time */}
      {result.processing_time && (
        <motion.div
          className="processing-time"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <span className="time-icon">⚡</span>
          <span>Analyzed in {result.processing_time.toFixed(2)}s</span>
        </motion.div>
      )}
    </motion.div>
  );
};

export default AnimatedResultCard;
