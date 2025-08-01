# Vietnamese Emotion Detection System - User Manual

*Complete user guide for the Vietnamese Emotion Detection System*

---

## Table of Contents

1. [Welcome to Vietnamese Emotion Detection](#welcome)
2. [Quick Start Guide](#quick-start)
3. [Dashboard Features](#dashboard-features)
4. [API Integration](#api-integration)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Welcome to Vietnamese Emotion Detection

### What You Can Do

The Vietnamese Emotion Detection System enables you to:

- **Real-time Analysis**: Analyze Vietnamese text instantly
- **Batch Processing**: Process multiple texts simultaneously
- **Multi-model Support**: Choose from LSTM, SVM, and other models
- **API Integration**: Integrate with your applications via REST API
- **Analytics Dashboard**: Monitor performance and usage metrics

### Supported Emotions

The system detects three primary emotions in Vietnamese text:

- **Positive** (Tích cực): Joy, happiness, excitement, satisfaction
- **Negative** (Tiêu cực): Sadness, anger, disappointment, frustration
- **Neutral** (Trung tính): Informational, objective, balanced content

---

## Quick Start Guide

### Step 1: Access the Dashboard

1. Open your web browser
2. Navigate to: `http://localhost:8501`
3. Wait for the dashboard to load

### Step 2: Enter Vietnamese Text

1. Find the text input area
2. Type or paste Vietnamese text
3. Example: "Hôm nay tôi rất vui vì được gặp bạn bè!"

### Step 3: Choose Your Model

1. Select from available models:
   - **Optimized LSTM**: Best accuracy (85.79%)
   - **SVM**: Fastest processing (30ms)
   - **Random Forest**: Balanced performance
   - **Logistic Regression**: Ultra-fast processing

### Step 4: Analyze

1. Click "Process" button
2. View results instantly
3. Examine confidence scores and probabilities

---

## Dashboard Features

### Single Text Analysis

```
┌─────────────────────────────────────────────────────────┐
│                    Emotion Detection                 │
├─────────────────────────────────────────────────────────┤
│ Enter Vietnamese text:                                  │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Hôm nay tôi rất vui vì được gặp bạn bè!            │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│  Model Selection: [LSTM ▼] [Analyze]               │
├─────────────────────────────────────────────────────────┤
│ Results:                                                │
│ ┌─────────────────┬──────────────┬────────────────────┐ │
│ │ Emotion: Positive                │
│ │ Confidence: 89.2%                │
│ │ Model Used: Optimized LSTM       │
│ │ Processing Time: 45ms            │
│ └─────────────────┴──────────────┴────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

#### How to Use Single Analysis

1. **Input Text**: Enter Vietnamese text (up to 1000 characters)
2. **Select Model**: Choose your preferred ML model
3. **Analyze**: Click the analyze button
4. **Review Results**: Examine emotion, confidence, and probabilities

#### Understanding Results

- **Emotion**: Predicted emotion category
- **Confidence**: Model's certainty (0-100%)
- **Probabilities**: Distribution across all emotions
- **Processing Time**: Analysis duration in milliseconds

### Batch Processing

#### Uploading Files

1. Click "Process" button
2. Select CSV file with Vietnamese text
3. Ensure file has 'text' column
4. Wait for processing to complete

#### File Format Requirements

```csv
text,additional_column
"Tôi yêu sản phẩm này!",metadata1
"Dịch vụ tệ quá",metadata2
"Bình thường thôi",metadata3
```

#### Processing Results

```
┌─────────────────────────────────────────────────────────┐
│ Batch Processing Results                                │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────┬─────────────┬─────────────┬─────────────┐│
│ │    Text     │   Emotion   │ Confidence  │    Time     ││
│ ├─────────────┼─────────────┼─────────────┼─────────────┤│
│ │ LSTM          │ Positive │   89.2%    │  45ms  │ Best││
│ │ PhoBERT       │ Positive │   91.5%    │ 120ms  │ Most││
│ │ SVM           │ Positive │   86.8%    │  30ms  │ Fast││
│ └─────────────┴─────────────┴─────────────┴─────────────┘│
└─────────────────────────────────────────────────────────┘
```

### Model Comparison

#### Comparing Multiple Models

1. Enter sample text
2. Click "Compare Models" 
3. View side-by-side results
4. Analyze performance differences

#### Interpretation Guide

- **Accuracy**: How often the model is correct
- **Speed**: Processing time per text
- **Use Case**: When to use each model

**Model Selection Guide:**
- **LSTM**: Use for highest accuracy requirements
- **SVM**: Use for high-volume, fast processing
- **Random Forest**: Use for balanced accuracy/speed
- **Logistic Regression**: Use for ultra-fast batch processing

---

## API Integration

### Authentication

Currently, the API operates without authentication for local deployment. For production deployment, implement API key authentication:

```python
headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}
```

### Base URL

- **Local Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com/api`

### Endpoints

#### Health Check

```http
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-01-15T10:30:00Z",
    "models_loaded": ["lstm", "svm", "random_forest", "logistic_regression"]
}
```

#### Single Prediction

```http
POST /predict
Content-Type: application/json

{
    "text": "Hôm nay tôi rất vui!",
    "model_type": "lstm"
}
```

**Response:**
```json
{
    "emotion": "positive",
    "confidence": 0.892,
    "probabilities": {
        "positive": 0.892,
        "negative": 0.068,
        "neutral": 0.040
    },
    "processing_time": 0.045,
    "model_used": "lstm"
}
```

#### Batch Prediction

```http
POST /predict/batch
Content-Type: application/json

{
    "texts": [
        "Tôi yêu sản phẩm này!",
        "Dịch vụ tệ quá",
        "Bình thường thôi"
    ],
    "model_type": "svm"
}
```

**Response:**
```json
{
    "predictions": [
        {
            "text": "Tôi yêu sản phẩm này!",
            "emotion": "positive",
            "confidence": 0.95
        },
        {
            "text": "Dịch vụ tệ quá",
            "emotion": "negative", 
            "confidence": 0.88
        },
        {
            "text": "Bình thường thôi",
            "emotion": "neutral",
            "confidence": 0.72
        }
    ],
    "processing_time": 0.132,
    "total_texts": 3
}
```

#### File Upload

```http
POST /predict/file
Content-Type: multipart/form-data

form-data:
  file: [your-csv-file]
  model_type: "lstm"
  text_column: "text"
```

### Error Handling

#### Common Error Responses

**400 Bad Request:**
```json
{
    "error": "validation_error",
    "message": "Text field is required",
    "details": {
        "field": "text",
        "constraint": "min_length: 1"
    }
}
```

**422 Unprocessable Entity:**
```json
{
    "error": "processing_error", 
    "message": "Unable to process text",
    "details": {
        "text_length": 5000,
        "max_length": 1000
    }
}
```

**500 Internal Server Error:**
```json
{
    "error": "server_error",
    "message": "Model prediction failed",
    "details": {
        "model": "lstm",
        "error_code": "CUDA_OUT_OF_MEMORY"
    }
}
```

### Integration Examples

#### Python Integration

```python
import requests
import json

class EmotionDetector:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
    
    def predict_emotion(self, text, model="lstm"):
        response = requests.post(
            f"{self.api_url}/predict",
            json={"text": text, "model_type": model}
        )
        return response.json()
    
    def predict_batch(self, texts, model="svm"):
        response = requests.post(
            f"{self.api_url}/predict/batch",
            json={"texts": texts, "model_type": model}
        )
        return response.json()

# Usage
detector = EmotionDetector()
result = detector.predict_emotion("Tôi rất hài lòng!")
print(result['emotion'])  # positive
```

#### JavaScript Integration

```javascript
class EmotionDetector {
    constructor(apiUrl = 'http://localhost:8000') {
        this.apiUrl = apiUrl;
    }
    
    async predictEmotion(text, model = 'lstm') {
        const response = await fetch(`${this.apiUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                model_type: model
            })
        });
        
        return await response.json();
    }
}

// Usage
const detector = new EmotionDetector();
detector.predictEmotion('Tôi rất vui!')
    .then(result => console.log(result.emotion));
```

#### cURL Examples

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hôm nay trời đẹp quá!",
    "model_type": "svm"
  }'
```

**Batch Prediction:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Tôi vui", "Tôi buồn", "Bình thường"],
    "model_type": "lstm"
  }'
```

---

## Best Practices

### Text Input Guidelines

#### Optimal Text Length
- **Minimum**: 5 characters
- **Optimal**: 20-200 characters  
- **Maximum**: 1000 characters
- **Recommendation**: Use complete sentences for best accuracy

#### Text Quality Tips
- **Use proper Vietnamese**: Avoid excessive slang or abbreviations
- **Complete thoughts**: Full sentences work better than fragments
- **Context matters**: Provide sufficient context for emotion detection
- **Avoid mixed languages**: Pure Vietnamese text yields best results

#### Examples of Good Input
```
✅ Good: "Tôi rất hài lòng với chất lượng sản phẩm này!"
✅ Good: "Dịch vụ khách hàng kém, tôi thất vọng."
✅ Good: "Phim này khá hay nhưng hơi dài."

❌ Avoid: "ok"
❌ Avoid: "123 abc xyz"
❌ Avoid: "This is English mixed với tiếng Việt"
```

### Model Selection Guide

#### When to Use Each Model

**Optimized LSTM (BiLSTM + Attention)**
- **Best for**: Highest accuracy requirements
- **Use when**: Quality is more important than speed
- **Examples**: Content moderation, research analysis
- **Performance**: 85.79% accuracy, ~45ms processing

**SVM (Support Vector Machine)**  
- **Best for**: Production environments with high volume
- **Use when**: Speed and consistency are priorities
- **Examples**: Real-time monitoring, live chat analysis
- **Performance**: 85.68% accuracy, ~30ms processing

**Random Forest**
- **Best for**: Balanced accuracy and interpretability
- **Use when**: You need to understand feature importance
- **Examples**: Analytics dashboards, A/B testing
- **Performance**: 84.23% accuracy, ~35ms processing

**Logistic Regression**
- **Best for**: Ultra-fast batch processing
- **Use when**: Processing thousands of texts quickly
- **Examples**: Bulk data analysis, ETL pipelines
- **Performance**: 83.45% accuracy, ~20ms processing

### Performance Optimization

#### For High-Volume Usage

1. **Use SVM for speed**: Fastest model with good accuracy
2. **Batch processing**: Process multiple texts together
3. **Caching**: Implement client-side caching for repeated texts
4. **Connection pooling**: Reuse HTTP connections

#### For High-Accuracy Usage

1. **Use Optimized LSTM**: Highest accuracy available
2. **Ensemble prediction**: Combine multiple model outputs
3. **Confidence filtering**: Use confidence thresholds
4. **Text preprocessing**: Clean input text properly

#### Memory Management

1. **Batch size limits**: Keep batches under 100 texts
2. **Text length limits**: Respect maximum length constraints
3. **Resource monitoring**: Monitor API response times
4. **Graceful degradation**: Handle model failures gracefully

### Dashboard Usage Tips

#### Navigation
- **Tabs**: Use tabs to switch between features
- **Sidebar**: Access settings and model information
- **Status indicators**: Monitor system health in real-time

#### Settings Configuration

Access through Settings menu:

```
Settings
├── Model Preferences
│   ├── Default Model: [LSTM ▼]
│   ├── Confidence Threshold: [70%]
│   └── Auto-refresh: [Enabled]
├── Display Options  
│   ├── Language: [English/Vietnamese]
│   ├── Theme: [Light/Dark]
│   └── Results Format: [Detailed/Simple]
└── Export Settings
    ├── Format: [CSV/JSON/Excel]
    ├── Include Probabilities: [Yes]
    └── Timestamp: [UTC/Local]
```

#### Keyboard Shortcuts

- **Ctrl+Enter**: Analyze current text
- **Ctrl+U**: Upload file
- **Ctrl+R**: Refresh results
- **Ctrl+S**: Save results
- **Ctrl+C**: Copy results to clipboard

---

## Troubleshooting

### Common Issues

#### "Model not loaded" Error

**Symptoms:** API returns model loading errors

**Solutions:**
1. Check if models exist in `/models` directory
2. Verify model file integrity
3. Restart the API service
4. Check available disk space (need 2GB+)
5. Verify Python dependencies are installed

**Command to fix:**
```bash
# Restart services
docker-compose restart

# Check model files
ls -la models/
```

#### Slow Response Times

**Symptoms:** API takes >5 seconds to respond

**Solutions:**
1. Switch to faster model (SVM or Logistic Regression)
2. Reduce text length
3. Use batch processing for multiple texts
4. Check server resources (CPU/memory)
5. Verify network connectivity

#### Dashboard Not Loading

**Symptoms:** Browser shows connection errors

**Solutions:**
1. Verify Streamlit service is running
2. Check port 8501 availability
3. Clear browser cache
4. Try different browser
5. Check firewall settings

#### Incorrect Predictions

**Symptoms:** Model gives unexpected results

**Solutions:**
1. Verify text is in Vietnamese
2. Check text quality (complete sentences)
3. Try different model for comparison
4. Examine confidence scores
5. Review text preprocessing

#### File Upload Failures

**Symptoms:** CSV upload doesn't work

**Solutions:**
1. Verify CSV format (must have 'text' column)
2. Check file size (<10MB recommended)
3. Ensure UTF-8 encoding
4. Remove special characters from headers
5. Try with smaller sample file

#### Memory Issues

**Symptoms:** "Out of memory" errors

**Solutions:**
1. Reduce batch size
2. Restart services to clear memory
3. Check available RAM
4. Use lighter models (Logistic Regression)
5. Process files in smaller chunks

### API Error Codes

#### Status Code Reference

- **200**: Success
- **400**: Bad Request (invalid input)
- **422**: Unprocessable Entity (validation error)
- **429**: Too Many Requests (rate limiting)
- **500**: Internal Server Error
- **503**: Service Unavailable

#### Error Response Format

```json
{
    "error": "error_type",
    "message": "Human readable message",
    "details": {
        "additional": "context information"
    },
    "timestamp": "2025-01-15T10:30:00Z"
}
```

### Performance Monitoring

#### Health Check Endpoint

Monitor system health using:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
    "status": "healthy",
    "models_loaded": 4,
    "memory_usage": "2.1GB",
    "uptime": "24:15:30"
}
```

#### Resource Monitoring

```bash
# Check CPU usage
top

# Check memory usage  
free -h

# Check disk space
df -h

# Check container status
docker ps
```

---

## Advanced Usage

### Custom Model Integration

#### Adding New Models

1. **Implement Model Class**: Extend base model interface
2. **Train Model**: Use provided training scripts
3. **Save Model**: Follow naming conventions
4. **Update Configuration**: Add model to config files
5. **Test Integration**: Verify API endpoints work

#### Model Format Requirements

```python
class CustomModel:
    def load_model(self, model_path):
        # Load your trained model
        pass
    
    def predict(self, text):
        # Return prediction in standard format
        return {
            'emotion': 'positive',
            'confidence': 0.85,
            'probabilities': {
                'positive': 0.85,
                'negative': 0.10,
                'neutral': 0.05
            }
        }
```

### Batch Processing Optimization

#### Large File Handling

For files >1000 rows:
1. **Split files**: Process in chunks of 500-1000 rows
2. **Use streaming**: Process line-by-line for memory efficiency
3. **Parallel processing**: Use multiple API calls
4. **Monitor progress**: Implement progress tracking
5. **Error handling**: Handle partial failures gracefully

#### Processing Pipeline Example

```python
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

def process_large_file(file_path, chunk_size=500):
    df = pd.read_csv(file_path)
    
    # Process in chunks
    results = []
    for chunk in df.groupby(df.index // chunk_size):
        chunk_data = chunk[1]
        
        # API call for chunk
        response = requests.post('/predict/batch', 
            json={'texts': chunk_data['text'].tolist()})
        
        results.extend(response.json()['predictions'])
    
    return results
```

---

## Support and Resources

### Documentation

- **API Documentation**: `/docs` endpoint (Swagger UI)
- **Model Documentation**: See individual model README files
- **Code Examples**: Available in `/examples` directory
- **Video Tutorials**: Available on project documentation site

### Community Support

- **GitHub Issues**: Report bugs and feature requests
- **Discussion Forum**: Community Q&A and best practices
- **Email Support**: technical-support@emotion-detection.com
- **Chat Support**: Real-time help during business hours

### Training and Workshops

- **Getting Started Workshop**: Monthly beginner sessions
- **Advanced Integration**: Quarterly technical deep-dives
- **Custom Training**: Available for enterprise customers
- **Certification Program**: Professional usage certification

---

**Happy Analyzing!**

*Vietnamese Emotion Detection System v1.0.0*  
*Built with ❤️ by Team InsideOut* 