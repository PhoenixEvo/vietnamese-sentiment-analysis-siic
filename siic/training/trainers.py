"""
Unified training interface for all models in SIIC.

This module provides a consistent training interface for:
- PhoBERT model training
- LSTM model training  
- Baseline model training
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from siic.utils.config import (
    MODELS_DIR, RESULTS_DIR, RANDOM_STATE, MODEL_CONFIGS
)
from siic.models.phobert import PhoBERTEmotionDetector
from siic.models.lstm import LSTMEmotionDetector
from siic.models.baselines import BaselineModels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedTrainer:
    """Unified trainer for all sentiment analysis models."""
    
    def __init__(self):
        self.supported_models = ['phobert', 'lstm', 'baselines']
        
    def train_phobert(self, **kwargs):
        """Train PhoBERT model."""
        logger.info("Training PhoBERT model...")
        
        detector = PhoBERTEmotionDetector(
            model_name=kwargs.get('model_name', 'vinai/phobert-base'),
            max_length=kwargs.get('max_length', 256)
        )
        
        # Prepare data
        df = detector.prepare_data()
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=RANDOM_STATE, stratify=df['label_id']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df['label_id']
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = detector.create_data_loaders(
            train_df, val_df, test_df, batch_size=kwargs.get('batch_size', 8)
        )
        
        # Train
        history = detector.train(
            train_loader, val_loader, 
            epochs=kwargs.get('epochs', 3),
            learning_rate=kwargs.get('learning_rate', 2e-5)
        )
        
        # Evaluate
        results = detector.evaluate(test_loader)
        
        # Save
        detector.save_model()
        
        return {'history': history, 'results': results}
    
    def train_lstm(self, **kwargs):
        """Train LSTM model."""
        logger.info("Training LSTM model...")
        
        detector = LSTMEmotionDetector(
            max_length=kwargs.get('max_length', 128)
        )
        
        # Load and prepare data
        from siic.utils.config import PROCESSED_DATA_DIR
        data_path = 'data/processed_uit_vsfc_data.csv'
        train_df, val_df, test_df = detector.prepare_data(data_path)
        
        # Create data loaders
        train_loader, val_loader, test_loader = detector.create_data_loaders(
            train_df, val_df, test_df, batch_size=kwargs.get('batch_size', 32)
        )
        
        # Create and train model
        detector.create_model()
        history = detector.train(
            train_loader, val_loader,
            epochs=kwargs.get('epochs', 15),
            learning_rate=kwargs.get('learning_rate', 0.001)
        )
        
        # Evaluate
        results = detector.evaluate(test_loader)
        
        # Save
        detector.save_model()
        
        return {'history': history, 'results': results}
    
    def train_baselines(self, **kwargs):
        """Train baseline models."""
        logger.info("Training baseline models...")
        
        baseline = BaselineModels()
        results = baseline.train_all_models()
        baseline.save_models()
        
        return {'results': results}
    
    def train(self, model_type: str, **kwargs) -> Dict[str, Any]:
        """Train specified model type."""
        if model_type not in self.supported_models:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        if model_type == 'phobert':
            return self.train_phobert(**kwargs)
        elif model_type == 'lstm':
            return self.train_lstm(**kwargs)
        elif model_type == 'baselines':
            return self.train_baselines(**kwargs)

def main():
    """CLI interface for training."""
    parser = argparse.ArgumentParser(description='Train emotion detection models')
    parser.add_argument('--model', choices=['phobert', 'lstm', 'baselines'], 
                       required=True, help='Model type to train')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_length', type=int, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Get model-specific defaults
    model_config = MODEL_CONFIGS.get(args.model, {})
    
    # Override with command line args
    train_kwargs = {}
    if args.epochs is not None:
        train_kwargs['epochs'] = args.epochs
    if args.batch_size is not None:
        train_kwargs['batch_size'] = args.batch_size  
    if args.learning_rate is not None:
        train_kwargs['learning_rate'] = args.learning_rate
    if args.max_length is not None:
        train_kwargs['max_length'] = args.max_length
    
    # Add model config defaults
    train_kwargs.update({k: v for k, v in model_config.items() if k not in train_kwargs})
    
    # Train model
    trainer = UnifiedTrainer()
    start_time = datetime.now()
    
    try:
        results = trainer.train(args.model, **train_kwargs)
        training_time = datetime.now() - start_time
        
        logger.info(f"Training completed in: {training_time}")
        logger.info(f"Results saved to: {RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 