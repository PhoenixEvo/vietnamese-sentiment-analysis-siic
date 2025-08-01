#!/usr/bin/env python3
"""
Optimize LSTM Model for higher accuracy (targeting 80-90%)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config.config import (EMOTION_LABELS, MODELS_DIR, PROCESSED_DATA_DIR,
                           RESULTS_DIR)
from src.models.lstm_model import LSTMEmotionDetector, VietnameseTextDataset


class ImprovedLSTMModel(nn.Module):
    """Improved LSTM model with BiLSTM and attention"""

    def __init__(
        self,
        vocab_size,
        embedding_dim=200,
        hidden_dim=256,
        num_layers=3,
        num_classes=3,
        dropout=0.5,
    ):
        super(ImprovedLSTMModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer with larger dimension
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # BiLSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.5)

        # Classification layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def attention_mechanism(self, lstm_output):
        """Apply attention mechanism"""
        # lstm_output: (batch_size, seq_len, hidden_dim * 2)
        attention_weights = torch.tanh(self.attention(lstm_output))  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # (batch_size, seq_len, 1)

        # Weighted sum
        attended_output = torch.sum(
            lstm_output * attention_weights, dim=1
        )  # (batch_size, hidden_dim * 2)
        return attended_output

    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)

        # Apply attention
        attended = self.attention_mechanism(lstm_out)  # (batch_size, hidden_dim * 2)

        # Dropout
        dropped = self.dropout1(attended)

        # First fully connected layer
        fc1_out = torch.relu(self.fc1(dropped))  # (batch_size, hidden_dim)

        # Batch normalization
        if fc1_out.size(0) > 1:  # Only apply batch norm if batch size > 1
            fc1_out = self.batch_norm(fc1_out)

        # Second dropout
        dropped2 = self.dropout2(fc1_out)

        # Final classification layer
        output = self.fc2(dropped2)  # (batch_size, num_classes)

        return output


class OptimizedLSTMDetector(LSTMEmotionDetector):
    """Optimized LSTM detector with improved training"""

    def create_model(self):
        """Create improved LSTM model"""
        self.model = ImprovedLSTMModel(
            vocab_size=len(self.vocab),
            embedding_dim=200,  # Larger embedding
            hidden_dim=256,  # Larger hidden dimension
            num_layers=3,  # More layers
            num_classes=len(self.emotion_labels),
            dropout=0.5,  # Higher dropout
        )

        self.model.to(self.device)

        print(
            f"Improved model created with {sum(p.numel() for p in self.model.parameters())} parameters"
        )
        return self.model

    def create_weighted_data_loaders(self, train_df, val_df, test_df, batch_size=32):
        """Create data loaders with weighted sampling for class imbalance"""

        # Calculate class weights
        labels = train_df["label_encoded"].values
        class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

        print(f"Class weights: {class_weights_dict}")

        # Create sample weights
        sample_weights = [class_weights_dict[label] for label in labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))

        # Create datasets
        train_dataset = VietnameseTextDataset(
            train_df["processed_text"], train_df["label_encoded"], self.vocab, self.max_length
        )

        val_dataset = VietnameseTextDataset(
            val_df["processed_text"], val_df["label_encoded"], self.vocab, self.max_length
        )

        test_dataset = VietnameseTextDataset(
            test_df["processed_text"], test_df["label_encoded"], self.vocab, self.max_length
        )

        # Create data loaders with weighted sampling for training
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, class_weights_dict

    def train_optimized(
        self, train_loader, val_loader, class_weights_dict, epochs=25, learning_rate=0.0005
    ):
        """Optimized training with better hyperparameters"""
        print("Starting optimized training...")

        # Weighted loss function for class imbalance
        class_weights_tensor = torch.tensor(
            [class_weights_dict[i] for i in range(len(class_weights_dict))], dtype=torch.float
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # AdamW optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        best_val_accuracy = 0
        best_model_state = None
        patience_counter = 0
        patience = 5

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # Training
            train_loss, train_acc = self.train_epoch(self.model, train_loader, criterion, optimizer)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            # Validation
            val_loss, val_acc = self.validate_epoch(self.model, val_loader, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Learning rate scheduling
            scheduler.step(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"New best validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Load best model
        self.model.load_state_dict(best_model_state)

        # Plot training history
        self.plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)

        return {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "best_val_accuracy": best_val_accuracy,
        }


def main():
    """Main optimization function"""
    print("LSTM Model Optimization for 80-90% Accuracy")
    print("=" * 60)

    # Initialize optimized detector
    detector = OptimizedLSTMDetector(max_length=256)  # Longer sequences

    # Prepare data
    data_path = "data/processed_uit_vsfc_data.csv"
    train_df, val_df, test_df = detector.prepare_data(data_path)

    print(f"\n📊 Data distribution:")
    for emotion in train_df["emotion"].value_counts().index:
        count = train_df["emotion"].value_counts()[emotion]
        percentage = count / len(train_df) * 100
        print(f"  {emotion}: {count} ({percentage:.1f}%)")

    # Create weighted data loaders
    train_loader, val_loader, test_loader, class_weights = detector.create_weighted_data_loaders(
        train_df, val_df, test_df, batch_size=64  # Larger batch size
    )

    # Create improved model
    detector.create_model()

    # Optimized training
    training_history = detector.train_optimized(
        train_loader,
        val_loader,
        class_weights,
        epochs=25,
        learning_rate=0.0005,  # Lower learning rate
    )

    # Evaluate model
    test_results = detector.evaluate(test_loader)

    # Save optimized model (just the state dict)
    optimized_path = os.path.join(MODELS_DIR, "optimized_lstm_emotion_model.pth")
    torch.save(detector.model.state_dict(), optimized_path)
    print(f"✅ Optimized model saved to: {optimized_path}")

    # Also save complete model with metadata for future loading
    complete_path = os.path.join(MODELS_DIR, "optimized_lstm_complete.pth")
    detector.save_model(complete_path)

    print("\n" + "=" * 60)
    print("🎉 OPTIMIZATION COMPLETE!")
    print(f"📊 Best Validation Accuracy: {training_history['best_val_accuracy']:.4f}")
    print(f"📊 Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"📊 Test F1 Score: {test_results['f1_score']:.4f}")

    if test_results["accuracy"] >= 0.8:
        print("TARGET ACHIEVED: 80%+ accuracy!")
    elif test_results["accuracy"] >= 0.7:
        print("📈 GOOD PROGRESS: 70%+ accuracy!")
    else:
        print("📈 IMPROVED but need more optimization")

    return training_history, test_results


if __name__ == "__main__":
    main()
