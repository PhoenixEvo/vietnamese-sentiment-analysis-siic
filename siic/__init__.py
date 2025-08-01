"""
SIIC - Vietnamese Sentiment Analysis System
==========================================

A comprehensive sentiment analysis system for Vietnamese text using 
multiple machine learning approaches including PhoBERT, LSTM, and traditional ML models.

Main modules:
- siic.data: Data loading and preprocessing
- siic.models: Model implementations (PhoBERT, LSTM, Baselines)
- siic.training: Training pipelines and optimizers
- siic.evaluation: Evaluation metrics and reporting
- siic.utils: Utilities and configuration
"""

__version__ = "1.0.0"
__author__ = "Team InsideOut"
__email__ = "team@insideout.com"

# Import only essential configs
from siic.utils.config import (
    EMOTION_LABELS, LABEL_TO_ID, PROCESSED_DATA_DIR, 
    MODELS_DIR, RESULTS_DIR, RANDOM_STATE
)

# Lazy imports - only import when needed to avoid dependency issues
# Use: from siic.models.phobert import PhoBERTEmotionDetector 