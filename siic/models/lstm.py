"""
LSTM Model for Vietnamese Emotion Detection
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from collections import Counter
import pickle
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Suppress warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# Suppress PyTorch warnings
logging.getLogger("torch").setLevel(logging.ERROR)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from siic.utils.config import (
    EMOTION_LABELS, LABEL_TO_ID, PROCESSED_DATA_DIR, 
    MODELS_DIR, RESULTS_DIR, RANDOM_STATE, MODEL_CONFIGS
)

class VietnameseTextDataset(Dataset):
    """Dataset class for Vietnamese text emotion classification"""
    
    def __init__(self, texts, labels, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx]
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        # Handle missing or invalid text values
        if pd.isna(text) or not isinstance(text, str):
            text = ""
        
        # Convert text to sequence of token indices
        tokens = str(text).split()
        sequence = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate sequence
        if len(sequence) < self.max_length:
            sequence.extend([self.vocab['<PAD>']] * (self.max_length - len(sequence)))
        else:
            sequence = sequence[:self.max_length]
            
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class LSTMEmotionClassifier(nn.Module):
    """LSTM model for emotion classification"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, 
                 num_layers=2, num_classes=4, dropout=0.3):
        super(LSTMEmotionClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state for classification
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Dropout
        dropped = self.dropout(last_hidden)
        
        # Classification
        output = self.classifier(dropped)
        
        return output

class CustomOptimizedLSTMModel(nn.Module):
    """Custom Optimized LSTM model to match saved state_dict structure"""
    
    def __init__(self, vocab_size, embedding_dim=200, hidden_dim=256, 
                 num_layers=2, num_classes=3, dropout=0.5):
        super(CustomOptimizedLSTMModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Multi-layer classification head (matching actual saved model)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # 512 -> 256
        self.fc2 = nn.Linear(hidden_dim, 128)             # 256 -> 128
        self.fc3 = nn.Linear(128, num_classes)            # 128 -> 3
        
        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)     # 256
        self.batch_norm2 = nn.BatchNorm1d(128)            # 128
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.5)
        
    def attention_mechanism(self, lstm_output):
        """Apply attention mechanism"""
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)
        return attended_output
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # BiLSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Apply attention
        attended = self.attention_mechanism(lstm_out)
        
        # Layer normalization
        attended = self.layer_norm(attended)
        
        # First layer: fc1 + batch_norm1 + dropout
        x = torch.relu(self.fc1(attended))
        if x.size(0) > 1:
            x = self.batch_norm1(x)
        x = self.dropout1(x)
        
        # Second layer: fc2 + batch_norm2 + dropout  
        x = torch.relu(self.fc2(x))
        if x.size(0) > 1:
            x = self.batch_norm2(x)
        x = self.dropout2(x)
        
        # Final classification layer
        logits = self.fc3(x)
        
        return logits

class ImprovedLSTMModel(nn.Module):
    """Improved LSTM model with BiLSTM and attention (for optimized model)"""
    
    def __init__(self, vocab_size, embedding_dim=200, hidden_dim=256, 
                 num_layers=3, num_classes=3, dropout=0.5):
        super(ImprovedLSTMModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer with larger dimension
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
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
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)  # (batch_size, hidden_dim * 2)
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

class LSTMEmotionDetector:
    """Main class for LSTM-based emotion detection"""
    
    def __init__(self, max_length=128):
        self.model = None
        self.vocab = None
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LABEL_TO_ID
        self.emotion_labels = EMOTION_LABELS
        

        
    def build_vocabulary(self, texts, min_freq=2):
        """Build vocabulary from texts"""
        print("Building vocabulary...")
        
        # Count word frequencies
        word_freq = Counter()
        for text in texts:
            tokens = str(text).split()
            word_freq.update(tokens)
        
        # Create vocabulary
        vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        
        for word, freq in word_freq.items():
            if freq >= min_freq:
                vocab[word] = idx
                idx += 1
                
        self.vocab = vocab
        print(f"Vocabulary size: {len(vocab)}")
        return vocab
    
    def prepare_data(self, data_path):
        """Prepare data for training"""
        print("Loading and preparing data...")
        
        # Load processed data
        df = pd.read_csv(data_path)
        
        # Encode labels
        df['label_encoded'] = df['emotion'].map(self.label_encoder)
        
        # Filter by split
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        val_df = df[df['split'] == 'validation'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
        
        print(f"Train size: {len(train_df)}")
        print(f"Validation size: {len(val_df)}")
        print(f"Test size: {len(test_df)}")
        
        # Build vocabulary from training data
        self.build_vocabulary(train_df['processed_text'])
        
        return train_df, val_df, test_df
    
    def create_data_loaders(self, train_df, val_df, test_df, batch_size=32):
        """Create PyTorch data loaders"""
        
        # Create datasets
        train_dataset = VietnameseTextDataset(
            train_df['processed_text'], train_df['label_encoded'], 
            self.vocab, self.max_length
        )
        
        val_dataset = VietnameseTextDataset(
            val_df['processed_text'], val_df['label_encoded'],
            self.vocab, self.max_length
        )
        
        test_dataset = VietnameseTextDataset(
            test_df['processed_text'], test_df['label_encoded'],
            self.vocab, self.max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def create_model(self):
        """Create LSTM model"""
        config = MODEL_CONFIGS['lstm']
        
        self.model = LSTMEmotionClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=len(self.emotion_labels),
            dropout=config['dropout']
        )
        
        self.model.to(self.device)
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct_predictions += (predicted == target).sum().item()
            total_predictions += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                correct_predictions += (predicted == target).sum().item()
                total_predictions += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=10, learning_rate=0.001):
        """Train the model"""
        print("Starting training...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_accuracy = 0
        best_model_state = None
        
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
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = self.model.state_dict().copy()
                print(f"New best validation accuracy: {val_acc:.4f}")
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        # Plot training history
        self.plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_accuracy
        }
    
    def plot_training_history(self, train_losses, train_accuracies, val_losses, val_accuracies):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'lstm_training_history.png'))
        plt.show()
    
    def evaluate(self, test_loader):
        """Evaluate the model"""
        print("Evaluating model...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        # Classification report
        report = classification_report(
            all_targets, all_predictions,
            target_names=list(self.emotion_labels.values()),
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            all_targets, all_predictions,
            target_names=list(self.emotion_labels.values())
        ))
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def predict_emotion(self, text):
        """Predict emotion for a single text"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        
        # Preprocess text
        tokens = str(text).split()
        sequence = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(sequence) < self.max_length:
            sequence.extend([self.vocab['<PAD>']] * (self.max_length - len(sequence)))
        else:
            sequence = sequence[:self.max_length]
        
        # Convert to tensor
        input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        emotion = self.emotion_labels[predicted_class]
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': {
                self.emotion_labels[i]: probabilities[0][i].item() 
                for i in range(len(self.emotion_labels))
            }
        }
    
    def save_model(self, filepath=None):
        """Save the trained model and vocabulary"""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, 'lstm_emotion_model.pth')
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state dict and vocabulary
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'model_config': {
                'vocab_size': len(self.vocab),
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'num_classes': len(self.emotion_labels)
            },
            'max_length': self.max_length
        }, filepath)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath=None):
        """Load a trained model"""
        if filepath is None:
            # Force load complete file only
            filepath = os.path.join(MODELS_DIR, 'improved_lstm_complete.pth')
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Complete LSTM model not found: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Check if this is a complete checkpoint or just state_dict
        if isinstance(checkpoint, dict) and 'vocab' in checkpoint:
            # Complete checkpoint with metadata
            self.vocab = checkpoint['vocab']
            config = checkpoint['model_config']
        else:
            # Only state_dict available - create fallback configuration

            
            # Build a default vocabulary (minimal for inference)
            try:
                # Try to load vocab from a separate file if exists
                vocab_path = os.path.join(MODELS_DIR, 'lstm_vocab.pkl')
                if os.path.exists(vocab_path):
                    with open(vocab_path, 'rb') as f:
                        self.vocab = pickle.load(f)
                    print("✅ Loaded vocabulary from separate file")
                else:
                    # Create vocab with correct size based on embedding layer
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    if 'embedding.weight' in state_dict:
                        vocab_size = state_dict['embedding.weight'].shape[0]
                        # Create minimal vocab with correct size
                        self.vocab = {'<PAD>': 0, '<UNK>': 1}
                        for i in range(2, vocab_size):
                            self.vocab[f'token_{i}'] = i

                    else:
                        self.vocab = {'<PAD>': 0, '<UNK>': 1}

            except:
                self.vocab = {'<PAD>': 0, '<UNK>': 1}

            
            # Infer model configuration from state_dict structure
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Checkpoint is the state_dict itself
                state_dict = checkpoint
            
            # Infer actual config from state_dict structure
            config = self._infer_model_config_from_state_dict(state_dict)

        
        # Create model with config
        
        # Get state_dict for model architecture detection
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Check if this is an optimized model (has attention layers in state dict)
        is_optimized = any('attention' in key for key in state_dict.keys())
        
        if is_optimized:
            # Create appropriate model based on state_dict structure
            if 'fc3.weight' in state_dict and 'layer_norm.weight' in state_dict:
        
                self.model = CustomOptimizedLSTMModel(
                    vocab_size=config['vocab_size'],
                    embedding_dim=config.get('embedding_dim', 200),
                    hidden_dim=config.get('hidden_dim', 256),
                    num_layers=config.get('num_layers', 3),
                    num_classes=config['num_classes'],
                    dropout=config.get('dropout', 0.5)
                )
            else:
        
                self.model = ImprovedLSTMModel(
                    vocab_size=config['vocab_size'],
                    embedding_dim=config.get('embedding_dim', 200),
                    hidden_dim=config.get('hidden_dim', 256),
                    num_layers=config.get('num_layers', 3),
                    num_classes=config['num_classes'],
                    dropout=config.get('dropout', 0.5)
                )
        else:
            # Create standard LSTMEmotionClassifier for regular models
    
            self.model = LSTMEmotionClassifier(
                vocab_size=config['vocab_size'],
                embedding_dim=config['embedding_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                num_classes=config['num_classes']
            )
        
        # Load model state with error handling
        try:
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print(f"⚠️ Error loading state_dict: {e}")
            print("Trying to load with strict=False...")
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load max_length
        if isinstance(checkpoint, dict):
            self.max_length = checkpoint.get('max_length', 128)
        else:
            self.max_length = 128
        
        # Ensure emotion_labels is set correctly
        if isinstance(checkpoint, dict) and 'emotion_labels' in checkpoint:
            self.emotion_labels = checkpoint['emotion_labels']
        else:
            # Use default emotion labels
            self.emotion_labels = EMOTION_LABELS
        


    def _infer_model_config_from_state_dict(self, state_dict):
        """Infer model configuration from state_dict structure"""
        config = {}
        
        # Infer vocab_size from embedding layer
        if 'embedding.weight' in state_dict:
            vocab_size, embedding_dim = state_dict['embedding.weight'].shape
            config['vocab_size'] = vocab_size
            config['embedding_dim'] = embedding_dim
            # print(f"Inferred vocab_size: {vocab_size}, embedding_dim: {embedding_dim}")
        else:
            config['vocab_size'] = 1000  # fallback
            config['embedding_dim'] = 200
        
        # Infer hidden_dim from LSTM layer
        if 'lstm.weight_ih_l0' in state_dict:
            # LSTM input-hidden weight has shape (4*hidden_size, input_size)
            weight_shape = state_dict['lstm.weight_ih_l0'].shape
            hidden_dim = weight_shape[0] // 4
            config['hidden_dim'] = hidden_dim
            # print(f"Inferred hidden_dim: {hidden_dim}")
        else:
            config['hidden_dim'] = 256
        
        # Count LSTM layers
        num_layers = 0
        for key in state_dict.keys():
            if key.startswith('lstm.weight_ih_l'):
                layer_num = int(key.split('_l')[1][0])
                num_layers = max(num_layers, layer_num + 1)
        config['num_layers'] = num_layers if num_layers > 0 else 2
        # print(f"Inferred num_layers: {config['num_layers']}")
        
        # Infer num_classes from final layer
        # Look for the final classification layer
        if 'fc3.weight' in state_dict:
            # Model has fc1, fc2, fc3 structure - fc3 is final classifier
            config['num_classes'] = state_dict['fc3.weight'].shape[0]
            # print(f"Inferred num_classes from fc3: {config['num_classes']}")
        elif 'fc2.weight' in state_dict:
            # Standard model with fc1, fc2 - fc2 is final classifier
            config['num_classes'] = state_dict['fc2.weight'].shape[0]
            # print(f"Inferred num_classes from fc2: {config['num_classes']}")
        elif 'classifier.weight' in state_dict:
            config['num_classes'] = state_dict['classifier.weight'].shape[0]
            # print(f"Inferred num_classes from classifier: {config['num_classes']}")
        else:
            config['num_classes'] = 3  # default: negative, neutral, positive
        
        # Default dropout
        config['dropout'] = 0.5
        
        return config

def main():
    """Main training function"""
    # Initialize LSTM detector
    lstm_detector = LSTMEmotionDetector(max_length=128)
    
    # Prepare data
    data_path = 'data/processed_uit_vsfc_data.csv'
    train_df, val_df, test_df = lstm_detector.prepare_data(data_path)
    
    # Create data loaders
    train_loader, val_loader, test_loader = lstm_detector.create_data_loaders(
        train_df, val_df, test_df, batch_size=32
    )
    
    # Create model
    lstm_detector.create_model()
    
    # Train model
    training_history = lstm_detector.train(
        train_loader, val_loader, epochs=15, learning_rate=0.001
    )
    
    # Evaluate model
    test_results = lstm_detector.evaluate(test_loader)
    
    # Save model
    lstm_detector.save_model()
    
    # Save results
    results = {
        'model_name': 'LSTM',
        'training_history': training_history,
        'test_results': test_results
    }
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save results as pickle
    with open(os.path.join(RESULTS_DIR, 'lstm_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*50)
    print("LSTM Training Complete!")
    print(f"Best Validation Accuracy: {training_history['best_val_accuracy']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test F1 Score: {test_results['f1_score']:.4f}")
    print("="*50)
    
    return results

if __name__ == "__main__":
    main()