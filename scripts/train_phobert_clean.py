#!/usr/bin/env python3
"""
Training script for PhoBERT emotion detection model (Clean version without emoji)
"""
import argparse
import logging
import os
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("phobert_training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from sklearn.model_selection import train_test_split

from config.config import RANDOM_STATE
from src.models.phobert_model import PhoBERTEmotionDetector


def main():
    parser = argparse.ArgumentParser(description="Train PhoBERT emotion detection model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument(
        "--model_name", type=str, default="vinai/phobert-base", help="PhoBERT model name"
    )

    args = parser.parse_args()

    logger.info("Starting PhoBERT training with parameters:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Max length: {args.max_length}")
    logger.info(f"  - Model: {args.model_name}")

    try:
        # Initialize detector
        logger.info("Initializing PhoBERT detector...")
        detector = PhoBERTEmotionDetector(model_name=args.model_name, max_length=args.max_length)

        # Prepare data
        logger.info("Loading and preparing data...")
        df = detector.prepare_data()

        # Split data
        logger.info("Splitting data...")
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=RANDOM_STATE, stratify=df["label_id"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df["label_id"]
        )

        logger.info(f"  - Train: {len(train_df)} samples")
        logger.info(f"  - Validation: {len(val_df)} samples")
        logger.info(f"  - Test: {len(test_df)} samples")

        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = detector.create_data_loaders(
            train_df, val_df, test_df, batch_size=args.batch_size
        )

        # Train model
        logger.info("Starting training...")
        start_time = datetime.now()

        history = detector.train(
            train_loader, val_loader, epochs=args.epochs, learning_rate=args.learning_rate
        )

        training_time = datetime.now() - start_time
        logger.info(f"Training completed in: {training_time}")
        logger.info(f"Best validation accuracy: {history['best_val_accuracy']:.4f}")

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        results = detector.evaluate(test_loader)

        # Save model
        logger.info("Saving model...")
        detector.save_model()

        # Save training results
        import pickle

        from config.config import RESULTS_DIR

        results_path = os.path.join(
            RESULTS_DIR, f"phobert_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        with open(results_path, "wb") as f:
            pickle.dump(
                {
                    **results,
                    "training_history": history,
                    "training_params": vars(args),
                    "training_time": str(training_time),
                },
                f,
            )

        logger.info(f"Results saved to: {results_path}")

        # Final summary
        logger.info("Training Summary:")
        logger.info(f"  - Final test accuracy: {results['accuracy']:.4f}")
        logger.info(f"  - Final test F1-score: {results['f1_score']:.4f}")
        logger.info(f"  - Training time: {training_time}")
        logger.info("PhoBERT training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
