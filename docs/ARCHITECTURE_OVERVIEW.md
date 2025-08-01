# 🏗️ SIIC - System Architecture Overview

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Model Architecture](#model-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Training Pipeline](#training-pipeline)
5. [Inference Pipeline](#inference-pipeline)
6. [Performance Analysis](#performance-analysis)
7. [Deployment Architecture](#deployment-architecture)
8. [Technology Stack](#technology-stack)

---

## 🎯 System Overview

**SIIC (Sentiment Intelligence and Insight Center)** là một hệ thống phân tích cảm xúc tiếng Việt sử dụng multiple model approaches để đạt được hiệu suất tối ưu.

### 🎯 Core Objectives
- **Accuracy**: Đạt >90% accuracy trên UIT-VSFC dataset
- **Robustness**: Support multiple model types để compare performance  
- **Scalability**: Dễ dàng thêm models mới và scale horizontally
- **Maintainability**: Clear separation of concerns và modular design

---

## 🤖 Model Architecture

### 🏆 Model Performance Ranking

| Rank | Model | Type | Accuracy | F1-Score | Characteristics |
|------|-------|------|----------|----------|----------------|
| 🥇 | **PhoBERT** | Transformer | **93.74%** | **93.44%** | Best overall, efficient training |
| 🥈 | **SVM** | Traditional ML | **85.61%** | **84.88%** | Surprisingly competitive |
| 🥉 | **LSTM** | Deep Learning | **85.02%** | **85.56%** | Good but overfitting issues |
| 4️⃣ | **Random Forest** | Traditional ML | **82.90%** | **81.93%** | Stable baseline |
| 5️⃣ | **Logistic Regression** | Traditional ML | **82.23%** | **83.42%** | Simple baseline |

### 🧠 Model Details

#### 1. **PhoBERT (Transformer) 🥇**
```python
Architecture:
├── PhoBERT Tokenizer (Vietnamese pre-trained)
├── PhoBERT Encoder (12 layers, 768 hidden size)
├── Classification Head (Dropout + Linear)
└── Output Layer (3 emotion classes)

Training:
- Epochs: 2 only
- Learning Rate: 2e-5
- Batch Size: 16
- No overfitting observed
```

#### 2. **LSTM (Deep Learning) 🥉**
```python
Architecture:
├── Embedding Layer (vocab_size=10000, embed_dim=128)
├── Bidirectional LSTM (hidden_size=128, num_layers=2)
├── Dropout Layer (rate=0.3)
└── Dense Layer (output=3 classes)

Training:
- Epochs: 30
- Learning Rate: 0.001
- Batch Size: 32
- Issue: Overfitting (train_acc=95%, val_acc=84%)
```

#### 3. **Traditional ML Models 🥈**
```python
SVM:
- Kernel: RBF
- C: 1.0
- Class Weight: Balanced
- Features: TF-IDF vectors

Random Forest:
- N Estimators: 100
- Class Weight: Balanced
- Features: TF-IDF vectors

Logistic Regression:
- C: 1.0
- Max Iter: 1000
- Class Weight: Balanced
- Features: TF-IDF vectors
```

---

## 📊 Data Pipeline

### 📥 Data Sources
- **Primary**: UIT-VSFC Dataset (Vietnamese Sentiment Dataset)
- **Format**: CSV with text and emotion labels
- **Classes**: 3 emotions (Negative, Neutral, Positive)

### 🔧 Preprocessing Pipeline
```python
Raw Text → Text Cleaning → Tokenization → Feature Engineering
    ↓              ↓              ↓              ↓
UIT-VSFC    Remove noise   Split words   TF-IDF/Embeddings
Dataset     Special chars  Normalize     PhoBERT tokens
```

### 🎯 Feature Engineering Strategies

#### 1. **For Traditional ML (TF-IDF)**
- Max features: 10,000
- N-grams: (1,2)
- Stop words: Vietnamese stop words
- Min/Max document frequency filtering

#### 2. **For LSTM (Word Embeddings)**
- Vocabulary size: 10,000
- Embedding dimension: 128
- Sequence length: 256
- Padding/Truncation strategies

#### 3. **For PhoBERT (Tokenization)**
- Pre-trained Vietnamese tokenizer
- Max length: 256
- Special tokens: [CLS], [SEP], [PAD]
- Subword tokenization

---

## 🚀 Training Pipeline

### 🔄 Training Workflow
```python
Data Loading → Preprocessing → Model Training → Evaluation → Model Saving
     ↓              ↓              ↓              ↓            ↓
Load CSV    Clean & Split   Train models   Calculate     Save to
UIT-VSFC    Train/Val/Test  Multiple types metrics       models/
```

### 📈 Training Strategies

#### **Traditional ML Training**
- **Quick Training**: No epochs, fit once
- **Cross Validation**: 5-fold for robust evaluation  
- **Hyperparameter Tuning**: Grid search for optimal params
- **Feature Selection**: TF-IDF optimization

#### **LSTM Training**
- **Epochs**: 30 with early stopping
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss
- **Regularization**: Dropout (0.3), Early stopping
- **Problem**: Overfitting detected at epoch 20+

#### **PhoBERT Training**
- **Epochs**: 2 only (very efficient!)
- **Optimizer**: AdamW (lr=2e-5)
- **Scheduler**: Linear warmup + decay
- **Regularization**: Built-in dropout in transformer layers
- **Advantage**: No overfitting, fast convergence

---

## 🔮 Inference Pipeline

### ⚡ Real-time Prediction Flow
```python
User Input → Preprocessing → Model Selection → Prediction → Post-processing → Response
    ↓             ↓              ↓              ↓              ↓              ↓
"Tôi rất    Text cleaning   Choose best    Run forward    Format result  JSON response
 vui hôm nay"  Tokenization   model (PhoBERT) pass         Add confidence  with emotion
```

### 🎯 Model Selection Strategy
1. **Default**: PhoBERT (best performance)
2. **Fast mode**: SVM (quick inference)
3. **Ensemble**: Combine multiple models for robustness

### 📊 Output Format
```json
{
  "emotion": "positive",
  "confidence": 0.94,
  "probabilities": {
    "negative": 0.02,
    "neutral": 0.04,
    "positive": 0.94
  },
  "model_used": "PhoBERT",
  "processing_time": "0.15s"
}
```

---

## 📈 Performance Analysis

### 🎯 Key Metrics

#### **Overall Performance**
- **Best Model**: PhoBERT (93.74% accuracy)
- **Baseline vs SOTA**: 10%+ improvement over traditional ML
- **Training Efficiency**: PhoBERT trains 15x faster than LSTM
- **Inference Speed**: All models <200ms per prediction

#### **Per-Class Performance** (PhoBERT)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 94.04% | 96.15% | 95.08% | 1,409 |
| Neutral | 64.79% | 43.81% | 52.27% | 164 |
| Positive | 95.14% | 95.85% | 95.50% | 1,590 |

#### **Training Insights**
- **PhoBERT**: Fast convergence, no overfitting
- **LSTM**: Overfitting after epoch 20
- **Traditional ML**: Quick training, competitive results
- **SVM**: Surprisingly good performance for simple model

### 🔍 Error Analysis
- **Neutral class**: Hardest to predict (imbalanced dataset)
- **Confusion**: Some negative/positive overlap
- **Context**: Long sentences sometimes misclassified

---

## 🚀 Deployment Architecture

### 🏗️ System Components

#### **Frontend Layer**
- **Web Dashboard**: Streamlit/Flask for visualization
- **REST API**: FastAPI for programmatic access
- **CLI Tools**: Python scripts for batch processing

#### **Application Layer**
- **Model Controller**: Model selection and loading
- **Emotion Predictor**: Real-time inference service
- **Model Trainer**: Training pipeline orchestration
- **Performance Evaluator**: Metrics computation and analysis

#### **Model Service Layer**
- **Model Loader**: Dynamic model loading and caching
- **Text Preprocessor**: Consistent text preprocessing
- **Result Processor**: Output formatting and confidence calculation

#### **Data Layer**
- **Data Loader**: Multi-format data ingestion
- **Data Validator**: Schema validation and quality checks
- **Feature Engineering**: Consistent feature extraction

#### **Storage Layer**
- **Model Storage**: Versioned model artifacts (models/)
- **Data Storage**: Raw and processed datasets (data/)
- **Results Storage**: Evaluation results and plots (results/)
- **Configuration**: Hyperparameters and settings (config/)

### 🐳 Infrastructure

#### **Containerization**
```dockerfile
# Multi-stage Docker build
Stage 1: Base dependencies (Python, PyTorch)
Stage 2: Model training environment
Stage 3: Production inference server
```

#### **Load Balancing**
- **Nginx**: Frontend proxy and load balancer
- **Multiple Workers**: Gunicorn/uWSGI for parallel processing
- **Model Caching**: Redis for model artifact caching

#### **Monitoring**
- **Metrics**: Prometheus for system metrics
- **Logging**: Structured logging with ELK stack
- **Alerting**: Model performance degradation alerts

---

## 💻 Technology Stack

### 🔧 Core Dependencies

#### **Machine Learning**
- **PyTorch**: Deep learning framework (LSTM, PhoBERT)
- **Transformers**: Hugging Face library (PhoBERT)
- **Scikit-learn**: Traditional ML algorithms
- **Pandas/NumPy**: Data manipulation and numerical computing

#### **NLP Specific**
- **PhoBERT**: Vietnamese pre-trained transformer
- **Tokenizers**: Fast tokenization libraries
- **NLTK/spaCy**: Text preprocessing utilities

#### **Web Framework**
- **FastAPI**: High-performance API framework
- **Streamlit**: Rapid dashboard development
- **Flask**: Lightweight web framework option

#### **Data & Storage**
- **CSV/JSON**: Data serialization formats
- **Pickle**: Model serialization
- **SQLite/PostgreSQL**: Optional database storage

#### **DevOps & Deployment**
- **Docker**: Containerization
- **Nginx**: Web server and reverse proxy
- **Git**: Version control
- **pytest**: Testing framework

### 📦 Project Structure
```
SIIC/
├── 📁 siic/                 # Core package
│   ├── 📁 data/            # Data handling modules
│   ├── 📁 models/          # Model implementations
│   ├── 📁 training/        # Training pipelines
│   ├── 📁 evaluation/      # Evaluation tools
│   └── 📁 utils/           # Utility functions
├── 📁 models/              # Saved model artifacts
├── 📁 data/                # Raw and processed data
├── 📁 results/             # Evaluation results
├── 📁 config/              # Configuration files
├── 📁 tests/               # Unit tests
├── 📁 docs/                # Documentation
├── 📁 api/                 # API server
├── 📁 dashboard/           # Web dashboard
└── 📁 scripts/             # Utility scripts
```

---

## 🎯 Future Improvements

### 🚀 Model Enhancements
1. **Ensemble Methods**: Combine multiple models for better accuracy
2. **Model Distillation**: Create lightweight models from PhoBERT
3. **Active Learning**: Iterative improvement with human feedback
4. **Multi-label Classification**: Support multiple emotions per text

### 🔧 System Improvements
1. **Real-time Training**: Online learning capabilities
2. **A/B Testing**: Model comparison in production
3. **Auto-scaling**: Dynamic resource allocation
4. **Edge Deployment**: Mobile/embedded inference

### 📊 Data Improvements
1. **Data Augmentation**: Synthetic data generation
2. **Domain Adaptation**: Support for different text domains
3. **Multilingual**: Support for other languages
4. **Streaming Data**: Real-time data ingestion

---

## 📞 Contact & Support

For questions about the architecture or implementation details:

- **Documentation**: `/docs` directory
- **Issues**: GitHub issues tracker
- **API Reference**: `/api/docs` endpoint
- **Performance Metrics**: `/results` directory

---

*Generated by SIIC Architecture Documentation System*
*Last Updated: January 2025* 