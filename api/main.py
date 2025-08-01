"""
Vietnamese Sentiment Analysis REST API
Production-ready FastAPI server for sentiment analysis
"""
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import torch
import uvicorn
import os
import sys
import json
import time
import io
from datetime import datetime
import logging

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from siic.models.lstm import LSTMEmotionDetector
from siic.models.baselines import BaselineModels
from siic.models.phobert import PhoBERTEmotionDetector
from siic.utils.config import EMOTION_LABELS, MODELS_DIR

# Initialize FastAPI app
app = FastAPI(
    title="Vietnamese Emotion Detection API",
    description="Production-ready API for Vietnamese social media emotion classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model instances
lstm_detector = None
baseline_models = None
phobert_detector = None

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str = Field(..., description="Vietnamese text to analyze", max_length=1000)
    model_type: Optional[str] = Field("phobert", description="Model type: 'phobert', 'lstm' or 'baseline'")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="List of Vietnamese texts to analyze")
    model_type: Optional[str] = Field("phobert", description="Model type: 'phobert', 'lstm' or 'baseline'")

class EmotionResponse(BaseModel):
    text: str
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time: float
    model_used: str

class BatchEmotionResponse(BaseModel):
    results: List[EmotionResponse]
    total_texts: int
    processing_time: float
    model_used: str

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    supported_emotions: List[str]
    max_text_length: int
    accuracy: Optional[float]
    f1_score: Optional[float]
    last_updated: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    system_info: Dict[str, Any]

# Startup event to load models
@app.on_event("startup")
async def load_models():
    """Load emotion detection models on startup"""
    global lstm_detector, baseline_models, phobert_detector
    
    logger.info("Loading models...")
    
    try:
        # Load PhoBERT model (Priority model)
        phobert_detector = PhoBERTEmotionDetector()
        phobert_model_path = os.path.join(MODELS_DIR, "phobert_emotion_model.pth")
        if os.path.exists(phobert_model_path):
            phobert_detector.load_model(phobert_model_path)
    
            # Debug: Check emotion labels after loading
            logger.info(f"🔍 PhoBERT emotion_labels: {phobert_detector.emotion_labels}")
            logger.info(f"🔍 PhoBERT label_encoder: {phobert_detector.label_encoder}")
        else:
            logger.warning("⚠️ PhoBERT model file not found")
            
        # Load LSTM model
        try:
            lstm_detector = LSTMEmotionDetector()
            # Let LSTM detector choose the best available model (complete -> improved -> standard)
            lstm_detector.load_model()
    
        except Exception as e:
            logger.warning(f"⚠️ Failed to load LSTM model: {str(e)}")
            lstm_detector = None
            
        # Load baseline models
        baseline_models = BaselineModels()
        baseline_models.load_models()

        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        # Continue startup even if models fail to load
        pass

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Vietnamese Emotion Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded={
            "phobert": phobert_detector is not None and phobert_detector.model is not None,
            "lstm": lstm_detector is not None and lstm_detector.model is not None,
            "baseline": baseline_models is not None and hasattr(baseline_models, 'models')
        },
        system_info={
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        }
    )

@app.post("/predict", response_model=EmotionResponse)
async def predict_emotion(input_data: TextInput):
    """Predict emotion for a single text"""
    start_time = time.time()
    
    try:
        # Validate input
        if not input_data.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Select model
        if input_data.model_type.lower() == "phobert":
            if phobert_detector is None or phobert_detector.model is None:
                raise HTTPException(status_code=503, detail="PhoBERT model not available")
            
            # Get prediction from PhoBERT
            result = phobert_detector.predict_emotion(input_data.text)
            model_used = "PhoBERT"
            
            # Debug: Log the result structure
            logger.info(f"🔍 PhoBERT result: {result}")
            logger.info(f"🔍 PhoBERT probabilities: {result.get('probabilities', {})}")
            for key, value in result.get('probabilities', {}).items():
                logger.info(f"🔍 Prob key '{key}' (type: {type(key)}): {value}")
            
        elif input_data.model_type.lower() == "lstm":
            if lstm_detector is None or lstm_detector.model is None:
                raise HTTPException(status_code=503, detail="LSTM model not available")
            
            # Get prediction from LSTM
            result = lstm_detector.predict_emotion(input_data.text)
            model_used = "Optimized LSTM"
            
        elif input_data.model_type.lower() == "baseline":
            if baseline_models is None:
                raise HTTPException(status_code=503, detail="Baseline models not available")
            
            # Get prediction from baseline (using best performing model)
            result = baseline_models.predict_emotion(input_data.text, model_name="svm")
            model_used = "SVM (Baseline)"
            
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Use 'phobert', 'lstm' or 'baseline'")
        
        processing_time = time.time() - start_time
        
        return EmotionResponse(
            text=input_data.text,
            emotion=result['emotion'],
            confidence=result['confidence'],
            probabilities=result.get('probabilities', {}),
            processing_time=processing_time,
            model_used=model_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchEmotionResponse)
async def predict_emotions_batch(input_data: BatchTextInput):
    """Predict emotions for multiple texts"""
    start_time = time.time()
    
    try:
        # Validate input
        if not input_data.texts:
            raise HTTPException(status_code=400, detail="Texts list cannot be empty")
        
        if len(input_data.texts) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 texts allowed per batch")
        
        results = []
        
        # Select model
        if input_data.model_type.lower() == "phobert":
            if phobert_detector is None or phobert_detector.model is None:
                raise HTTPException(status_code=503, detail="PhoBERT model not available")
            model_used = "PhoBERT"
            
        elif input_data.model_type.lower() == "lstm":
            if lstm_detector is None or lstm_detector.model is None:
                raise HTTPException(status_code=503, detail="LSTM model not available")
            model_used = "Optimized LSTM"
            
        elif input_data.model_type.lower() == "baseline":
            if baseline_models is None:
                raise HTTPException(status_code=503, detail="Baseline models not available")
            model_used = "SVM (Baseline)"
            
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Use 'phobert', 'lstm' or 'baseline'")
        
        # Process each text
        for text in input_data.texts:
            text_start = time.time()
            
            if not text.strip():
                # Handle empty text
                result_item = EmotionResponse(
                    text=text,
                    emotion="neutral",
                    confidence=0.0,
                    probabilities={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                    processing_time=0.0,
                    model_used=model_used
                )
            else:
                # Get prediction
                if input_data.model_type.lower() == "phobert":
                    result = phobert_detector.predict_emotion(text)
                elif input_data.model_type.lower() == "lstm":
                    result = lstm_detector.predict_emotion(text)
                else:
                    result = baseline_models.predict_emotion(text, model_name="svm")
                
                text_processing_time = time.time() - text_start
                
                result_item = EmotionResponse(
                    text=text,
                    emotion=result['emotion'],
                    confidence=result['confidence'],
                    probabilities=result.get('probabilities', {}),
                    processing_time=text_processing_time,
                    model_used=model_used
                )
            
            results.append(result_item)
        
        total_processing_time = time.time() - start_time
        
        return BatchEmotionResponse(
            results=results,
            total_texts=len(input_data.texts),
            processing_time=total_processing_time,
            model_used=model_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/file")
async def predict_from_file(
    file: UploadFile = File(...),
    model_type: str = "phobert",
    text_column: str = "text"
):
    """Predict emotions from uploaded CSV file"""
    try:
        # Validate file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate column
        if text_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
            )
        
        # Limit rows
        if len(df) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 rows allowed")
        
        # Prepare batch prediction
        texts = df[text_column].fillna("").astype(str).tolist()
        batch_input = BatchTextInput(texts=texts, model_type=model_type)
        
        # Get predictions
        batch_result = await predict_emotions_batch(batch_input)
        
        # Create result DataFrame
        result_df = df.copy()
        result_df['predicted_emotion'] = [r.emotion for r in batch_result.results]
        result_df['confidence'] = [r.confidence for r in batch_result.results]
        
        # Save to temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"emotion_predictions_{timestamp}.csv"
        output_path = f"/tmp/{output_filename}"
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type='text/csv'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in file prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")

@app.get("/models", response_model=List[ModelInfo])
async def get_model_info():
    """Get information about available models"""
    models = []
    
    # Convert EMOTION_LABELS dict to list
    supported_emotions_list = list(EMOTION_LABELS.values())
    
    # PhoBERT model info (Best performing)
    if phobert_detector is not None and phobert_detector.model is not None:
        models.append(ModelInfo(
            model_name="PhoBERT",
            model_type="transformer",
            supported_emotions=supported_emotions_list,
            max_text_length=256,
            accuracy=93.74,
            f1_score=93.44,
            last_updated="2025-01-16"
        ))
    
    # LSTM model info
    if lstm_detector is not None and lstm_detector.model is not None:
        models.append(ModelInfo(
            model_name="Optimized LSTM",
            model_type="deep_learning",
            supported_emotions=supported_emotions_list,
            max_text_length=128,
            accuracy=85.79,
            f1_score=85.47,
            last_updated="2025-01-15"
        ))
    
    # Baseline models info
    if baseline_models is not None:
        models.append(ModelInfo(
            model_name="SVM Classifier",
            model_type="traditional_ml",
            supported_emotions=supported_emotions_list,
            max_text_length=1000,
            accuracy=85.68,
            f1_score=84.93,
            last_updated="2025-01-15"
        ))
    
    return models

@app.get("/models/{model_name}/performance")
async def get_model_performance(model_name: str):
    """Get detailed performance metrics for a specific model"""
    performance_data = {
        "phobert": {
            "accuracy": 93.74,
            "f1_score": 93.44,
            "precision": {"positive": 0.95, "negative": 0.94, "neutral": 0.65},
            "recall": {"positive": 0.96, "negative": 0.96, "neutral": 0.44},
            "confusion_matrix": [[1073, 10, 33], [33, 46, 26], [35, 15, 1156]],
            "training_epochs": 2,
            "parameters": "~135M",
            "model_size": "543MB"
        },
        "lstm": {
            "accuracy": 85.79,
            "f1_score": 85.47,
            "precision": {"positive": 0.83, "negative": 0.89, "neutral": 0.0},
            "recall": {"positive": 0.83, "negative": 0.89, "neutral": 0.0},
            "confusion_matrix": [[1347, 167, 0], [268, 1330, 0], [16, 46, 0]],
            "training_epochs": 21,
            "parameters": "~4.5M"
        },
        "svm": {
            "accuracy": 85.68,
            "f1_score": 84.93,
            "precision": {"positive": 0.85, "negative": 0.88, "neutral": 0.0},
            "recall": {"positive": 0.82, "negative": 0.90, "neutral": 0.0},
            "kernel": "RBF",
            "parameters": "Support vectors"
        }
    }
    
    model_key = model_name.lower().replace(" ", "_").replace("optimized_", "")
    
    if model_key not in performance_data:
        raise HTTPException(status_code=404, detail="Model performance data not found")
    
    return performance_data[model_key]

# Background task for model retraining (placeholder)
@app.post("/models/retrain")
async def retrain_model(background_tasks: BackgroundTasks, model_type: str = "lstm"):
    """Trigger model retraining (placeholder for future implementation)"""
    if model_type not in ["phobert", "lstm", "baseline"]:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    # Add background task for retraining
    background_tasks.add_task(placeholder_retrain_task, model_type)
    
    estimated_times = {
        "phobert": "60-120 minutes",
        "lstm": "30-60 minutes",
        "baseline": "10-20 minutes"
    }
    
    return {
        "message": f"Model retraining initiated for {model_type}",
        "status": "queued",
        "estimated_time": estimated_times[model_type]
    }

async def placeholder_retrain_task(model_type: str):
    """Placeholder function for model retraining"""
    logger.info(f"Starting retraining for {model_type} model...")
    # Implementation would go here
    logger.info(f"Retraining completed for {model_type} model")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    ) 