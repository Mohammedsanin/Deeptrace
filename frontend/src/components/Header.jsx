import './Header.css'

function Header() {
    return (
        <header className="header">
            <div className="container">
                <div className="header-content">
                    <div className="logo">
                        <div className="shield-icon">
                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 2L4 6V12C4 16.55 7.16 20.74 12 22C16.84 20.74 20 16.55 20 12V6L12 2Z"
                                    fill="url(#gradient)" stroke="currentColor" strokeWidth="2" />
                                <defs>
                                    <linearGradient id="gradient" x1="4" y1="2" x2="20" y2="22">
                                        <stop offset="0%" stopColor="#667eea" />
                                        <stop offset="100%" stopColor="#764ba2" />
                                    </linearGradient>
                                </defs>
                            </svg>
                        </div>
                        <div className="logo-text">
                            <h1 className="gradient-text">Deeptrace</h1>
                            <p className="tagline">AI-Powered Deepfake Detection</p>
                        </div>
                    </div>

                    <div className="header-stats">
                        <div className="stat">
                            <span className="stat-value gradient-text">99.2%</span>
                            <span className="stat-label">Accuracy</span>
                        </div>
                        <div className="stat">
                            <span className="stat-value gradient-text">Real-time</span>
                            <span className="stat-label">Analysis</span>
                        </div>
                    </div>
                </div>
            </div>
        </header>
    )
}

export default Header
