# Hướng dẫn Upload Project lên GitHub

## Bước 1: Tạo Repository trên GitHub

1. Truy cập [GitHub.com](https://github.com) và đăng nhập
2. Click vào nút "New" hoặc "+" để tạo repository mới
3. Đặt tên repository: `SIIC-Emotion-Analysis` hoặc `vietnamese-sentiment-analysis`
4. Chọn "Public" hoặc "Private" tùy ý
5. **KHÔNG** check vào "Initialize this repository with a README" (vì đã có sẵn)
6. Click "Create repository"

## Bước 2: Kết nối Local Repository với GitHub

Sau khi tạo repository trên GitHub, bạn sẽ thấy các lệnh để kết nối. Chạy các lệnh sau:

```bash
# Thêm remote origin (thay thế URL bằng URL repository của bạn)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Đổi tên branch chính thành 'main' (GitHub mặc định)
git branch -M main

# Push code lên GitHub
git push -u origin main
```

## Bước 3: Kiểm tra Repository

1. Truy cập repository trên GitHub
2. Kiểm tra các file đã được upload đúng
3. Kiểm tra file `.gitignore` đã loại trừ các file không cần thiết

## Các file đã được loại trừ bởi .gitignore:

### Thư mục:
- `venv/` - Virtual environment
- `__pycache__/` - Python cache files
- `docs/_build/` - Documentation build files
- `results/` - Kết quả training và evaluation
- `outputs/` - Output files
- `checkpoints/` - Model checkpoints

### File lớn:
- `*.csv` - Data files (emotion_analysis_results.csv, youtube_comments.csv)
- `*.pkl` - Pickle files (models)
- `*.pth` - PyTorch model files
- `*.h5` - HDF5 files
- `models/phobert_tokenizer/` - Tokenizer files

### File nhạy cảm:
- `.env` - Environment variables
- `config/secrets.py` - Secret configurations
- `config/local.py` - Local configurations

## Bước 4: Cập nhật README (Tùy chọn)

Sau khi upload, bạn có thể cập nhật README.md để thêm:
- Badge cho repository
- Link đến live demo (nếu có)
- Screenshots của dashboard
- Contributing guidelines

## Bước 5: Tạo GitHub Actions (Tùy chọn)

Tạo file `.github/workflows/ci.yml` để tự động test:

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest
    - name: Run tests
      run: |
        pytest tests/
```

## Lưu ý quan trọng:

1. **File lớn**: Các file model (.pth, .pkl) và data (.csv) đã được loại trừ để tránh làm repository quá nặng
2. **Environment**: File venv/ đã được loại trừ, người dùng khác sẽ tự tạo virtual environment
3. **Secrets**: Các file chứa thông tin nhạy cảm đã được loại trừ
4. **Documentation**: Tất cả docs/ đã được include để người khác có thể hiểu project

## Troubleshooting:

### Nếu gặp lỗi "fatal: remote origin already exists":
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### Nếu file lớn vẫn được track:
```bash
git rm --cached <file_name>
git commit -m "Remove large files"
git push
```

### Nếu muốn thêm file đã bị ignore:
```bash
git add -f <file_name>
git commit -m "Add ignored file"
git push
``` 