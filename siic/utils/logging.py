"""
Logging utilities for SIIC Vietnamese Sentiment Analysis System.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(name: str, level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_training_logger(model_name: str) -> logging.Logger:
    """Get logger for training with timestamped log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"artifacts/logs/training_{model_name}_{timestamp}.log"
    return setup_logger(f"siic.training.{model_name}", log_file=log_file)

def get_evaluation_logger() -> logging.Logger:
    """Get logger for evaluation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"artifacts/logs/evaluation_{timestamp}.log"
    return setup_logger("siic.evaluation", log_file=log_file) 