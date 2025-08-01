# 🚀 Quick Start Guide

Hướng dẫn nhanh để chạy project SIIC Vietnamese Sentiment Analysis

## 📋 Yêu cầu hệ thống

- **Python**: 3.8 hoặc cao hơn
- **RAM**: Tối thiểu 8GB (16GB khuyến nghị)
- **GPU**: Khuyến nghị cho training (không bắt buộc)
- **OS**: Windows, Linux, macOS

## ⚡ Cài đặt nhanh

### 1. Clone và setup
```bash
# Clone repository
git clone https://github.com/PhoenixEvo/vietnamese-sentiment-analysis-siic.git
cd vietnamese-sentiment-analysis-siic

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Kiểm tra cài đặt
```bash
# Chạy test đơn giản
python -c "import siic; print('✅ Installation successful!')"

# Hoặc chạy tests
pytest tests/ -v
```

## 🎯 Sử dụng nhanh

### **Chạy Dashboard (Khuyến nghị cho người mới)**
```bash
# Chạy Streamlit dashboard
streamlit run dashboard/app.py

# Truy cập: http://localhost:8501
```

### **Chạy API Server**
```bash
# Chạy FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Truy cập: http://localhost:8000/docs
```

### **Test với dữ liệu mẫu**
```python
# Tạo file test_data.py
import pandas as pd

# Tạo dữ liệu mẫu
sample_data = {
    'text': [
        'Sản phẩm rất tốt, tôi rất thích!',
        'Dịch vụ tệ quá, không nên mua',
        'Bình thường, không có gì đặc biệt'
    ],
    'label': ['positive', 'negative', 'neutral']
}

df = pd.DataFrame(sample_data)
df.to_csv('data/sample_data.csv', index=False)
print("✅ Sample data created!")
```

## 🔧 Training Models

### **1. Train PhoBERT (Mô hình tốt nhất)**
```bash
# Train PhoBERT model
python scripts/train_phobert_clean.py

# Thời gian: ~30-60 phút (tùy GPU)
# Kết quả: models/phobert_emotion_model.pth
```

### **2. Train LSTM**
```bash
# Train LSTM model
python scripts/optimize_lstm_improved.py

# Thời gian: ~15-30 phút
# Kết quả: models/improved_lstm_emotion_model.pth
```

### **3. Train Baseline Models**
```bash
# Train SVM, Random Forest, Logistic Regression
python scripts/evaluate_models.py

# Kết quả: models/*.pkl files
```

## 📊 Evaluation

### **Chạy comprehensive evaluation**
```bash
# Đánh giá tất cả models
python scripts/comprehensive_evaluation.py

# Kết quả: results/comprehensive_evaluation_report.md
```

### **So sánh models**
```bash
# So sánh PhoBERT với baselines
python scripts/compare_phobert_results.py

# Kết quả: results/comparison_phobert.png
```

## 🎮 Sử dụng trong code

```python
# Import models
from siic.models.phobert import PhoBERTSentimentAnalyzer
from siic.models.lstm import LSTMSentimentAnalyzer
from siic.models.baselines import BaselineSentimentModels

# Sử dụng PhoBERT
phobert_model = PhoBERTSentimentAnalyzer()
result = phobert_model.predict("Tôi rất thích sản phẩm này!")
print(f"Sentiment: {result}")

# Sử dụng LSTM
lstm_model = LSTMSentimentAnalyzer()
result = lstm_model.predict("Dịch vụ tệ quá!")
print(f"Sentiment: {result}")

# Sử dụng baselines
baseline_models = BaselineSentimentModels()
result = baseline_models.predict_svm("Sản phẩm bình thường")
print(f"Sentiment: {result}")
```

## 🐛 Troubleshooting

### **Lỗi thường gặp:**

#### 1. **ModuleNotFoundError**
```bash
# Giải pháp: Cài đặt lại package
pip install -e .
```

#### 2. **CUDA out of memory**
```bash
# Giải pháp: Giảm batch size
# Trong scripts/train_phobert_clean.py, thay đổi:
# batch_size = 4  # thay vì 8
```

#### 3. **File not found errors**
```bash
# Tạo thư mục cần thiết
mkdir -p data models results
```

#### 4. **PhoBERT download issues**
```bash
# Tải thủ công PhoBERT
python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base'); model = AutoModel.from_pretrained('vinai/phobert-base')"
```

## 📁 Cấu trúc project

```
vietnamese-sentiment-analysis-siic/
├── siic/                    # Core package
│   ├── models/             # Model implementations
│   ├── data/               # Data processing
│   ├── training/           # Training utilities
│   └── evaluation/         # Evaluation metrics
├── scripts/                # CLI scripts
├── dashboard/              # Streamlit app
├── api/                    # FastAPI backend
├── tests/                  # Test files
├── docs/                   # Documentation
└── data/                   # Data files (tạo thủ công)
```

## 🎯 Next Steps

1. **Chạy dashboard** để xem giao diện
2. **Train PhoBERT model** để có kết quả tốt nhất
3. **Thử nghiệm với dữ liệu của bạn**
4. **Đóng góp** qua Pull Requests

## 📞 Hỗ trợ

- **Issues**: [GitHub Issues](https://github.com/PhoenixEvo/vietnamese-sentiment-analysis-siic/issues)
- **Documentation**: Xem thư mục `docs/`
- **Examples**: Xem thư mục `scripts/`

---

**Chúc bạn thành công! 🚀** 