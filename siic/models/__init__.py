"""
Model implementations for Vietnamese emotion detection.

Available models:
- PhoBERTModel: Transformer-based model (93.74% accuracy)
- LSTMModel: BiLSTM with attention (85.02% accuracy)  
- BaselineModels: Traditional ML models (SVM, RF, LR)
"""

# Lazy imports to avoid heavy dependencies on import
# Use specific imports like: from siic.models.phobert import PhoBERTEmotionDetector 