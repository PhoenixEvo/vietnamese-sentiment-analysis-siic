# Contributing to SIIC

Cảm ơn bạn đã quan tâm đến việc đóng góp cho dự án SIIC! Dưới đây là hướng dẫn để bạn có thể đóng góp một cách hiệu quả.

## Cách đóng góp

### 1. Báo cáo lỗi (Bug Reports)

Nếu bạn tìm thấy lỗi, vui lòng:

1. Kiểm tra xem lỗi đã được báo cáo chưa trong [Issues](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/issues)
2. Tạo issue mới với:
   - Mô tả chi tiết lỗi
   - Các bước để tái tạo lỗi
   - Thông tin môi trường (OS, Python version, etc.)
   - Screenshot nếu có thể

### 2. Đề xuất tính năng (Feature Requests)

1. Kiểm tra xem tính năng đã được đề xuất chưa
2. Tạo issue với label "enhancement"
3. Mô tả chi tiết tính năng và lý do cần thiết

### 3. Đóng góp code

#### Thiết lập môi trường phát triển

```bash
# Fork repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
pip install -e .

# Cài đặt development dependencies
pip install pytest pytest-cov flake8 black isort bandit safety
```

#### Quy trình làm việc

1. **Tạo branch mới**
   ```bash
   git checkout -b feature/your-feature-name
   # hoặc
   git checkout -b fix/your-bug-fix
   ```

2. **Thực hiện thay đổi**
   - Tuân thủ coding standards
   - Viết tests cho tính năng mới
   - Cập nhật documentation nếu cần

3. **Chạy tests**
   ```bash
   pytest tests/
   pytest tests/ --cov=siic --cov-report=html
   ```

4. **Chạy linting**
   ```bash
   flake8 siic/ scripts/
   black siic/ scripts/
   isort siic/ scripts/
   ```

5. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

6. **Push và tạo Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Python Code Style

- Tuân thủ PEP 8
- Sử dụng Black cho code formatting
- Sử dụng isort cho import sorting
- Độ dài dòng tối đa: 88 ký tự (Black default)

### Commit Messages

Sử dụng [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new sentiment analysis model
fix: resolve memory leak in LSTM training
docs: update README with installation guide
test: add unit tests for PhoBERT model
refactor: optimize data preprocessing pipeline
```

### Documentation

- Viết docstrings cho tất cả functions và classes
- Cập nhật README.md nếu thêm tính năng mới
- Thêm comments cho code phức tạp

## Testing

### Viết tests

```python
# tests/test_models.py
import pytest
from siic.models.phobert import PhoBERTSentimentAnalyzer

def test_phobert_model_initialization():
    model = PhoBERTSentimentAnalyzer()
    assert model is not None

def test_phobert_prediction():
    model = PhoBERTSentimentAnalyzer()
    result = model.predict("Tôi rất thích sản phẩm này!")
    assert result in ['positive', 'negative', 'neutral']
```

### Chạy tests

```bash
# Chạy tất cả tests
pytest

# Chạy tests với coverage
pytest --cov=siic --cov-report=html

# Chạy tests cụ thể
pytest tests/test_models.py::test_phobert_model_initialization
```

## Review Process

1. **Code Review**: Tất cả PR sẽ được review bởi maintainers
2. **CI/CD**: PR phải pass tất cả tests và linting checks
3. **Documentation**: Cập nhật docs nếu cần thiết
4. **Merge**: Sau khi được approve, PR sẽ được merge

## Cấu trúc Project

```
siic/
├── siic/                    # Core package
│   ├── data/               # Data loading và preprocessing
│   ├── models/             # Model implementations
│   ├── training/           # Training utilities
│   ├── evaluation/         # Evaluation metrics
│   └── utils/              # Utility functions
├── scripts/                # CLI scripts
├── tests/                  # Test files
├── docs/                   # Documentation
└── examples/               # Usage examples
```

## Liên hệ

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/discussions)
- **Email**: your-email@example.com

## License

Bằng cách đóng góp, bạn đồng ý rằng đóng góp của bạn sẽ được cấp phép theo cùng license với project (MIT License).

## Cảm ơn

Cảm ơn tất cả contributors đã đóng góp cho SIIC! 🎉 