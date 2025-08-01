# SIIC - Vietnamese Sentiment Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Hệ thống phân tích cảm xúc (sentiment analysis) cho bình luận mạng xã hội tiếng Việt, sử dụng các mô hình NLP hiện đại như PhoBERT, LSTM và các mô hình học máy truyền thống.

## Project Overview
- **Mục tiêu**: Phân tích sentiment (tích cực, tiêu cực, trung tính) trong bình luận mạng xã hội tiếng Việt.
- **Ứng dụng**: Giám sát thương hiệu, phân tích phản hồi khách hàng, hỗ trợ chăm sóc khách hàng tự động.
- **Nhóm phát triển**: InsideOut
  - **Leader**: Nguyễn Nhật Phát
  - **Member**: Nguyễn Tiến Huy

## Features
- **Multi-model Sentiment Analysis**: PhoBERT, LSTM, SVM, Random Forest, Logistic Regression
- **Vietnamese NLP Pipeline**: Làm sạch, tiền xử lý, vector hóa tối ưu cho tiếng Việt
- **Web Dashboard**: Giao diện Streamlit trực quan, realtime
- **REST API**: FastAPI backend cho tích hợp hệ thống
- **Báo cáo & Đánh giá**: So sánh, trực quan hóa kết quả các mô hình

## Model Performance
| Model               | Accuracy | F1-Score |
|---------------------|----------|----------|
| PhoBERT             | 93.74%   | 93.44%   |
| SVM                 | 85.61%   | 84.88%   |
| Optimized LSTM      | 85.02%   | 85.56%   |
| Random Forest       | 82.90%   | 81.93%   |
| Logistic Regression | 82.23%   | 83.42%   |

## Project Structure
```
SIIC/
├── siic/                    # Core package: data, models, training, evaluation, utils
├── scripts/                 # CLI scripts: train, evaluate, dashboard, batch
├── dashboard/               # Streamlit web app
├── api/                     # FastAPI backend
├── models/                  # Trained model files
├── data/                    # Datasets (raw, processed)
├── artifacts/               # Logs, reports, plots
├── configs/                 # Model & training configs
└── tests/                   # Unit & integration tests
```

## Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd SIIC

# Cài đặt môi trường ảo (khuyến nghị)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt package
pip install -e .
# Hoặc chỉ cài dependencies
pip install -r requirements.txt
```

### Usage

#### Windows Batch Commands
```batch
scripts\siic.bat test         # Kiểm tra cài đặt
scripts\siic.bat dashboard   # Mở dashboard
scripts\siic.bat train-baselines
scripts\siic.bat train-lstm
scripts\siic.bat train-phobert
scripts\siic.bat evaluate
```

#### Python CLI
```bash
# Training
python scripts/train.py --model phobert --epochs 3 --batch_size 8
python scripts/train.py --model lstm --epochs 15 --batch_size 32
python scripts/train.py --model baselines

# Evaluation
python scripts/evaluate.py --comprehensive
python scripts/evaluate.py --generate-report

# Dashboard
python scripts/dashboard.py --port 8501

# API Server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Using in Code
```python
# Import models
from siic.models.phobert import PhoBERTSentimentAnalyzer
from siic.models.lstm import LSTMSentimentAnalyzer
from siic.models.baselines import BaselineSentimentModels

# Configuration
from siic.utils.config import EMOTION_LABELS, MODELS_DIR

# Data processing
from siic.data.loaders import load_uit_vsfc_data
from siic.data.preprocessors import VietnameseTextPreprocessor
```

## Dataset
- **Nguồn**: UIT-VSFC (Vietnamese Social Media Feedback Corpus)
- **File dữ liệu**: data/processed_uit_vsfc_data.csv
- **Nhãn**: Negative (0), Neutral (1), Positive (2)
- **Ngôn ngữ**: Tiếng Việt

## Sentiment Labels
- **0**: Negative (Tiêu cực)
- **1**: Neutral (Trung tính)
- **2**: Positive (Tích cực)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Contact
- **Team**: InsideOut
- **Email**: team@insideout.com
- **Project**: Vietnamese Sentiment Analysis System 