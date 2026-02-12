# 🛡️ Deeptrace - End-to-End Deepfake Detection System

A comprehensive AI-powered deepfake detection system that analyzes both audio and video content using deep learning models. Features a modern React frontend and Python FastAPI backend.

![Deeptrace](https://img.shields.io/badge/AI-Deepfake%20Detection-purple)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![React](https://img.shields.io/badge/React-18.2-61dafb)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)

## 🌟 Features

- **Audio Deepfake Detection**: Detects AI-generated or manipulated audio using CNN-based analysis
- **Video Deepfake Detection**: Identifies face manipulation in videos using CNN+LSTM architecture
- **Real-time Analysis**: Fast inference with confidence scores
- **Modern UI**: Beautiful, responsive interface with glassmorphism design
- **End-to-End Pipeline**: Complete workflow from data preprocessing to deployment

## 🏗️ Architecture

```
┌─────────────────┐
│  React Frontend │
│   (Vite + CSS)  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  FastAPI Backend│
│   (Python 3.9+) │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Preprocessing  │
│  Audio / Video  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Deep Learning  │
│  CNN / LSTM     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   Prediction    │
│  Real / Fake    │
└─────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- Node.js 16 or higher
- GPU (recommended) or CPU
- Kaggle API credentials (for dataset download)

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up Kaggle API** (for dataset download):
   - Create account on [Kaggle](https://www.kaggle.com/)
   - Go to Account → API → Create New API Token
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

## 📊 Dataset Preparation

### Download Datasets

Run the dataset download script:

```bash
cd backend
python datasets/download_datasets.py
```

This will download:
- **ASVspoof 2019** (Audio) - Voice spoofing detection dataset
- **Deepfake Videos** (Video) - Face manipulation dataset

### Dataset Structure

After download, your data directory will look like:

```
backend/data/
├── asvspoof2019/
│   └── LA/
│       ├── ASVspoof2019_LA_train/
│       ├── ASVspoof2019_LA_dev/
│       └── ASVspoof2019_LA_eval/
└── deepfake_videos/
    ├── real/
    └── fake/
```

## 🏋️ Model Training

### Train Audio Model

```bash
cd backend
python train_audio.py
```

**Training Configuration**:
- Batch size: 32
- Learning rate: 0.001
- Epochs: 50 (with early stopping)
- Validation split: 20%

**Model Architecture**:
- 4 Convolutional blocks with batch normalization
- Global average pooling
- Fully connected layers with dropout
- Binary classification (Real/Fake)

### Train Video Model

```bash
cd backend
python train_video.py
```

**Training Configuration**:
- Batch size: 4 (reduced for video)
- Learning rate: 0.001
- Epochs: 50 (with early stopping)
- Validation split: 20%

**Model Architecture**:
- EfficientNet-B0 CNN backbone (pretrained)
- Bidirectional LSTM for temporal analysis
- Attention mechanism
- Binary classification (Real/Fake)

### Training Tips

- **GPU Recommended**: Training on GPU is significantly faster
- **Start Small**: Test with fewer epochs first (e.g., `--epochs 2`)
- **Monitor Progress**: Training logs show loss and accuracy metrics
- **Early Stopping**: Training stops automatically if validation loss doesn't improve

## 🚀 Running the Application

### Start Backend Server

```bash
cd backend
python main.py
```

Server will start at: `http://localhost:8000`

**API Endpoints**:
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict/audio` - Audio deepfake detection
- `POST /predict/video` - Video deepfake detection

### Start Frontend

```bash
cd frontend
npm run dev
```

Frontend will open at: `http://localhost:3000`

### Using the Application

1. Open browser to `http://localhost:3000`
2. Upload an audio or video file (drag & drop or click to browse)
3. Click "Analyze for Deepfake"
4. View results with confidence scores

## 📁 Project Structure

```
end to end/
├── backend/
│   ├── config.py                 # Configuration settings
│   ├── main.py                   # FastAPI application
│   ├── requirements.txt          # Python dependencies
│   ├── train_audio.py           # Audio model training
│   ├── train_video.py           # Video model training
│   ├── datasets/
│   │   ├── download_datasets.py # Dataset downloader
│   │   ├── audio_preprocessing.py
│   │   └── video_preprocessing.py
│   ├── models/
│   │   ├── audio_model.py       # Audio CNN model
│   │   └── video_model.py       # Video CNN+LSTM model
│   ├── inference/
│   │   ├── audio_inference.py   # Audio prediction
│   │   └── video_inference.py   # Video prediction
│   ├── data/                    # Datasets (gitignored)
│   ├── saved_models/            # Trained models (gitignored)
│   └── uploads/                 # Temporary uploads (gitignored)
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── App.css
        ├── index.css            # Design system
        └── components/
            ├── Header.jsx
            ├── Header.css
            ├── UploadSection.jsx
            ├── UploadSection.css
            ├── ResultCard.jsx
            └── ResultCard.css
```

## 🔬 Technical Details

### Audio Processing Pipeline

1. Load audio file (mono, 16kHz)
2. Extract Mel spectrogram (128 bins)
3. Normalize features
4. Pad/truncate to fixed length (4 seconds)
5. Feed to CNN model
6. Output: Real/Fake probability

### Video Processing Pipeline

1. Extract frames at 5 FPS
2. Detect faces using MediaPipe
3. Crop and resize faces (224×224)
4. Create temporal sequences (30 frames)
5. Feed to CNN+LSTM model
6. Output: Real/Fake probability

### Model Performance

Expected performance (after full training):
- **Audio Model**: ~95% accuracy on ASVspoof dataset
- **Video Model**: ~90% accuracy on deepfake videos

*Note: Actual performance depends on training duration and dataset size*

## 🎨 UI Features

- **Glassmorphism Design**: Modern frosted glass effect
- **Gradient Accents**: Vibrant purple and pink gradients
- **Smooth Animations**: Framer Motion powered transitions
- **Drag & Drop**: Easy file upload
- **Responsive**: Works on all screen sizes
- **Dark Mode**: Optimized for dark backgrounds

## 🛠️ API Usage

### Example: Audio Detection

```bash
curl -X POST "http://localhost:8000/predict/audio" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_audio.wav"
```

**Response**:
```json
{
  "success": true,
  "type": "audio",
  "filename": "sample_audio.wav",
  "result": "Real",
  "confidence": 0.95,
  "probabilities": {
    "Real": 0.95,
    "Deepfake": 0.05
  }
}
```

### Example: Video Detection

```bash
curl -X POST "http://localhost:8000/predict/video" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_video.mp4"
```

## ⚙️ Configuration

Edit `backend/config.py` to customize:

- Model hyperparameters
- Audio/video preprocessing settings
- API settings (port, file size limits)
- Training configuration

## 🐛 Troubleshooting

### Models Not Loading

**Issue**: "Model not found" error

**Solution**: Train the models first using `train_audio.py` and `train_video.py`

### Dataset Download Fails

**Issue**: Kaggle API error

**Solution**: 
1. Ensure `kaggle.json` is in correct location
2. Accept dataset terms on Kaggle website
3. Check internet connection

### CUDA Out of Memory

**Issue**: GPU memory error during training

**Solution**:
1. Reduce batch size in `config.py`
2. Use CPU instead: Set `DEVICE = "cpu"` in `config.py`
3. Process fewer frames for video

### Frontend Can't Connect to Backend

**Issue**: CORS or connection error

**Solution**:
1. Ensure backend is running on port 8000
2. Check CORS settings in `main.py`
3. Verify frontend API URL in `App.jsx`

## 📝 License

This project is for educational purposes. Please ensure compliance with dataset licenses when using for commercial purposes.

## 🙏 Acknowledgments

- **ASVspoof 2019**: Audio spoofing detection dataset
- **Deepfake Videos Dataset**: Video manipulation dataset
- **PyTorch**: Deep learning framework
- **FastAPI**: Modern Python web framework
- **React**: Frontend library

## 📧 Support

For issues or questions, please check:
1. This README
2. Code comments
3. Configuration files

---

**Built with ❤️ using Python, PyTorch, FastAPI, and React**
