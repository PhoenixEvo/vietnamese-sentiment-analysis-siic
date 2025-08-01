# Vietnamese Sentiment Analysis System - User Manual

*Comprehensive user guide for the Vietnamese Sentiment Analysis System*

---

## Table of Contents
1. Introduction
2. Installation Guide
3. Dashboard Usage
4. API Usage
5. Model Retraining
6. Model Evaluation
7. Troubleshooting
8. FAQ
9. Support Information

---

## 1. Introduction
The Vietnamese Sentiment Analysis System enables users to analyze the sentiment of Vietnamese text using multiple machine learning models (LSTM, PhoBERT, SVM, Random Forest, Logistic Regression). The system features an interactive dashboard, RESTful API, and is easily deployable via Docker.

**Main Components:**
- Web Dashboard (Streamlit)
- API Backend (Flask/FastAPI)
- Pre-trained models
- Docker for packaging and deployment

**Supported Sentiments:**
- Positive (Tích cực)
- Negative (Tiêu cực)
- Neutral (Trung tính)

---

## 2. Installation Guide
### 2.1. Docker Installation
1. Install Docker Desktop.
2. Clone the repository:
   ```bash
   git clone <repo_url>
   cd SIIC
   ```
3. Build and run the containers:
   ```bash
   docker-compose up --build
   ```
4. Access the dashboard at: http://localhost:8501

### 2.2. Manual Installation (Without Docker)
1. Install Python 3.8+
2. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r dashboard/requirements.txt
   ```
3. Run the API and dashboard as described in the README.md

---

## 3. Dashboard Usage
- Open your browser and go to http://localhost:8501
- Enter Vietnamese text in the input box.
- Select the desired model (LSTM, PhoBERT, SVM, etc.).
- Click "Analyze Sentiment" to get the prediction.
- View the sentiment result, confidence score, and visualizations.
- Optionally, upload a CSV file for batch analysis (ensure the file has a 'text' column).
- Download results if supported.

**Dashboard Features:**
- Single text analysis with model selection
- Batch processing via file upload
- Model comparison and performance visualization

---

## 4. API Usage
### 4.1. Endpoints
- **Health Check:**
  ```http
  GET /health
  ```
- **Single Prediction:**
  ```http
  POST /predict
  Content-Type: application/json
  {
    "text": "Hôm nay tôi rất vui!",
    "model_type": "lstm"
  }
  ```
- **Batch Prediction:**
  ```http
  POST /predict/batch
  Content-Type: application/json
  {
    "texts": ["Tôi yêu sản phẩm này!", "Dịch vụ tệ quá", "Bình thường thôi"],
    "model_type": "svm"
  }
  ```
- **File Upload:**
  ```http
  POST /predict/file
  Content-Type: multipart/form-data
  file: [your-csv-file]
  model_type: "lstm"
  text_column: "text"
  ```

### 4.2. Example Usage
- **Python:**
  ```python
  import requests
  response = requests.post(
      'http://localhost:8000/predict',
      json={"text": "Tôi rất hài lòng!", "model_type": "lstm"}
  )
  print(response.json())
  ```
- **cURL:**
  ```bash
  curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "Hôm nay trời đẹp quá!", "model_type": "svm"}'
  ```

---

## 5. Model Retraining
- To retrain models with new data:
  - LSTM: `python scripts/train_lstm.py`
  - PhoBERT: `python scripts/train_phobert_clean.py`
- Trained models are saved in the `models/` directory.
- You can adjust training parameters in the scripts or config files.

---

## 6. Model Evaluation
- Run evaluation scripts:
  - `python scripts/evaluate.py` to assess model performance.
  - View accuracy, F1-score, and confusion matrix in the terminal or output files.
- Use notebooks in the `notebooks/` directory for result visualization.

---

## 7. Troubleshooting
- **Docker installation issues:** Check Docker Desktop version and permissions.
- **Missing dependencies:** Ensure all requirements are installed in the correct environment.
- **Dashboard not running:** Check logs and ensure port 8501 is available.
- **API not responding:** Verify models are loaded and input data is correctly formatted.
- **File upload errors:** Ensure CSV files have a 'text' column and are UTF-8 encoded.

---

## 8. FAQ
- **Can I use the system on MacOS?** Yes, with Docker or compatible Python.
- **Can I retrain with my own data?** Yes, replace the data file and rerun the training scripts.
- **Can I add new models?** Yes, extend the model modules and update the dashboard/API accordingly.
- **What is the maximum text length?** Recommended up to 1000 characters per input.

---

## 9. Support Information
- Email: <team email or maintainer>
- Zalo/Telegram: <support group link>
- GitHub Issues: <repository link>

---

*Happy Analyzing!*

*Vietnamese Sentiment Analysis System v1.0.0* 