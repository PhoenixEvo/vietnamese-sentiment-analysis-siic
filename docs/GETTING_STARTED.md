# 🚀 Getting Started with SIIC

Welcome to the SIIC (Vietnamese Emotion Detection System)! This guide will help you get up and running quickly.

## 📋 Prerequisites

- Python 3.8+ installed
- Git installed
- 4GB+ RAM (8GB+ recommended for training)
- CUDA-compatible GPU (optional, for faster training)

## ⚡ Quick Setup

### 1. Clone & Install
```bash
# Clone the repository
git clone <repository-url>
cd SIIC

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install package
pip install -e .
```

### 2. Verify Installation
```bash
# Test imports (Windows)
scripts\siic.bat test

# Test imports (Manual)
python -c "from siic.models.baselines import BaselineModels; print('✅ Setup OK!')"
```

## 🎯 First Steps

### 1. Launch Dashboard
```bash
# Windows (Recommended)
scripts\siic.bat dashboard

# Manual
python scripts/dashboard.py
```
Visit: `http://localhost:8501`

### 2. Test Existing Models
The repository comes with pre-trained models. Test them in the dashboard:
- Enter Vietnamese text: "Tôi rất vui hôm nay!"
- Click "Predict Emotion"
- See results from all models

### 3. Train Your First Model
```bash
# Train baseline models (fastest, ~2 minutes)
scripts\siic.bat train-baselines

# Train LSTM model (~10 minutes)
scripts\siic.bat train-lstm

# Train PhoBERT model (~30 minutes, requires GPU)
scripts\siic.bat train-phobert
```

## 🏗️ Project Structure Overview

```
SIIC/
├── siic/                    # 🎯 Main package
│   ├── models/             # Model implementations
│   ├── data/               # Data processing
│   ├── training/           # Training pipelines
│   └── evaluation/         # Evaluation tools
├── scripts/                # 🛠️ CLI commands
├── dashboard/              # 🌐 Web interface
├── api/                   # 🔌 REST API
├── models/                # 💾 Trained models
└── data/                  # 📊 Dataset
```

## 🎮 Common Commands

### Windows Batch Commands (Easiest)
```batch
scripts\siic.bat help              # Show all commands
scripts\siic.bat test              # Test installation
scripts\siic.bat dashboard         # Launch dashboard
scripts\siic.bat train-baselines   # Train baseline models
scripts\siic.bat evaluate          # Run evaluation
scripts\siic.bat clean             # Clean build files
```

### Python Commands (Advanced)
```bash
# Training specific models
python scripts/train.py --model phobert --epochs 2 --batch_size 8
python scripts/train.py --model lstm --epochs 10 --batch_size 32

# Evaluation
python scripts/evaluate.py --comprehensive
python scripts/evaluate.py --generate-report

# Dashboard with custom port
python scripts/dashboard.py --port 8502 --dev
```

## 🔍 Using Models in Code

### Basic Usage
```python
from siic.models.phobert import PhoBERTEmotionDetector
from siic.utils.config import EMOTION_LABELS

# Load model
detector = PhoBERTEmotionDetector()
detector.load_model()

# Predict emotion
text = "Tôi rất hạnh phúc hôm nay!"
prediction = detector.predict(text)
emotion = EMOTION_LABELS[prediction]
print(f"Emotion: {emotion}")
```

### Advanced Usage
```python
from siic.models.lstm import LSTMEmotionDetector
from siic.models.baselines import BaselineModels
from siic.data.preprocessors import VietnameseTextPreprocessor

# Multiple models comparison
phobert = PhoBERTEmotionDetector()
lstm = LSTMEmotionDetector()  
baselines = BaselineModels()

# Load all models
phobert.load_model()
lstm.load_model()
baselines.load_models()

# Predict with all models
text = "Sản phẩm này thật tệ!"
results = {
    'PhoBERT': phobert.predict(text),
    'LSTM': lstm.predict(text),
    'SVM': baselines.predict_svm(text)
}
```

## 📈 Performance Expectations

| Model | Training Time | Accuracy | Use Case |
|-------|---------------|----------|----------|
| Baseline Models | 2 minutes | 82-86% | Quick prototyping |
| LSTM | 10 minutes | 85% | Balanced performance |
| PhoBERT | 30 minutes | 94% | Best accuracy |

## 🔧 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

**CUDA/GPU Issues**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only version
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Memory Issues**
- Reduce batch size: `--batch_size 4`
- Use smaller models: Train baselines first
- Close other applications

**Port Already in Use**
```bash
# Use different port
python scripts/dashboard.py --port 8502
```

## 📚 Next Steps

1. **Explore Models**: Try different models in the dashboard
2. **Custom Training**: Train on your own Vietnamese text data
3. **API Integration**: Use the REST API in your applications
4. **Extend Models**: Add new emotion categories or models
5. **Production**: Deploy using Docker in production

## 🆘 Getting Help

- **Check Documentation**: `docs/` folder
- **Run Tests**: `pytest tests/ -v`
- **Check Issues**: Look for similar problems
- **Contact Team**: team@insideout.com

---

**🎉 You're all set!** Start with the dashboard and explore the different models. Happy emotion detection! 🚀 