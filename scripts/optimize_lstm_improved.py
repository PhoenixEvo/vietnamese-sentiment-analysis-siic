#!/usr/bin/env python3
"""
Improved LSTM Model - Target 85.79% accuracy
Cải thiện từ version trước để đạt performance như report
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config.config import PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, EMOTION_LABELS

class VietnameseTextDataset(Dataset):
    """Dataset for Vietnamese text classification"""
    
    def __init__(self, texts, labels, vocab, max_length=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        # Tokenize and convert to indices
        tokens = text.split()
        indices = [self.vocab.get(token, self.vocab.get('<UNK>', 0)) for token in tokens]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class ImprovedLSTMModel(nn.Module):
    """Improved LSTM model with better regularization"""
    
    def __init__(self, vocab_size, embedding_dim=200, hidden_dim=256, 
                 num_layers=2, num_classes=3, dropout=0.3):
        super(ImprovedLSTMModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)
        
        # BiLSTM layers with layer normalization
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.5)
        
        # Classification layers với residual connection
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Batch normalization
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def attention_mechanism(self, lstm_output, mask=None):
        """Apply attention mechanism with masking"""
        # lstm_output: (batch_size, seq_len, hidden_dim * 2)
        attention_weights = torch.tanh(self.attention(lstm_output))  # (batch_size, seq_len, 1)
        
        # Apply mask if provided
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        
        attention_weights = torch.softmax(attention_weights, dim=1)  # (batch_size, seq_len, 1)
        
        # Weighted sum
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)  # (batch_size, hidden_dim * 2)
        return attended_output
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Create attention mask
        mask = (x != 0).float()  # Mask for padding tokens
        
        # Embedding with dropout
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        
        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Apply attention with masking
        attended = self.attention_mechanism(lstm_out, mask)  # (batch_size, hidden_dim * 2)
        
        # Dropout
        dropped = self.dropout1(attended)
        
        # First fully connected layer
        fc1_out = torch.relu(self.fc1(dropped))  # (batch_size, hidden_dim)
        
        # Batch normalization
        if fc1_out.size(0) > 1:
            fc1_out = self.batch_norm1(fc1_out)
        
        # Second dropout
        dropped2 = self.dropout2(fc1_out)
        
        # Second fully connected layer
        fc2_out = torch.relu(self.fc2(dropped2))  # (batch_size, hidden_dim // 2)
        
        # Batch normalization
        if fc2_out.size(0) > 1:
            fc2_out = self.batch_norm2(fc2_out)
        
        # Final classification layer
        output = self.fc3(fc2_out)  # (batch_size, num_classes)
        
        return output

class ImprovedLSTMDetector:
    """Improved LSTM detector with better training strategy"""
    
    def __init__(self, max_length=256):
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.vocab = None
        self.model = None
        
    def create_vocabulary(self, texts, min_freq=3):
        """Create vocabulary from texts with higher min_freq"""
        word_freq = Counter()
        for text in texts:
            tokens = str(text).split()
            word_freq.update(tokens)
        
        # Create vocabulary
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, freq in word_freq.items():
            if freq >= min_freq:
                vocab[word] = len(vocab)
        
        print(f"Vocabulary size: {len(vocab)} (min_freq={min_freq})")
        return vocab
    
    def prepare_data(self, data_path):
        """Prepare and split data with stratified sampling"""
        df = pd.read_csv(data_path)
        
        # Remove very short texts
        df = df[df['processed_text'].str.len() > 10]
        
        # Encode labels
        le = LabelEncoder()
        df['label_encoded'] = le.fit_transform(df['emotion'])
        
        # Stratified split to ensure balanced distribution
        X = df['processed_text']
        y = df['label_encoded']
        
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Second split: 15% val, 15% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Create DataFrames
        train_df = pd.DataFrame({
            'processed_text': X_train, 
            'label_encoded': y_train, 
            'emotion': le.inverse_transform(y_train)
        })
        val_df = pd.DataFrame({
            'processed_text': X_val, 
            'label_encoded': y_val, 
            'emotion': le.inverse_transform(y_val)
        })
        test_df = pd.DataFrame({
            'processed_text': X_test, 
            'label_encoded': y_test, 
            'emotion': le.inverse_transform(y_test)
        })
        
        # Create vocabulary
        self.vocab = self.create_vocabulary(X_train, min_freq=3)
        
        return train_df, val_df, test_df
    
    def create_model(self):
        """Create improved LSTM model"""
        self.model = ImprovedLSTMModel(
            vocab_size=len(self.vocab),
            embedding_dim=200,
            hidden_dim=256,
            num_layers=2,  # Reduced layers to prevent overfitting
            num_classes=len(self.emotion_labels),
            dropout=0.3    # Reduced dropout
        )
        
        self.model.to(self.device)
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def create_data_loaders(self, train_df, val_df, test_df, batch_size=32):
        """Create data loaders with balanced sampling"""
        
        # Calculate class weights
        labels = train_df['label_encoded'].values
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print(f"Class weights: {class_weights_dict}")
        
        # Create sample weights for balanced sampling
        sample_weights = [class_weights_dict[label] for label in labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
        
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, class_weights_dict
    
    def train_improved(self, train_loader, val_loader, class_weights_dict, 
                      epochs=30, learning_rate=0.001):
        """Improved training with cosine annealing"""
        print("Starting improved training...")
        
        # Weighted loss function
        class_weights_tensor = torch.tensor(
            [class_weights_dict[i] for i in range(len(class_weights_dict))], 
            dtype=torch.float
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Adam optimizer with weight decay
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0001)
        
        # Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training history
        history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }
        
        best_val_accuracy = 0
        best_model_state = None
        patience_counter = 0
        patience = 7  # Increased patience
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            history['train_losses'].append(train_loss)
            history['train_accuracies'].append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            history['val_losses'].append(val_loss)
            history['val_accuracies'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"✅ New best validation accuracy: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation accuracy: {best_val_accuracy:.4f}")
        
        history['best_val_accuracy'] = best_val_accuracy
        return history
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train one epoch with gradient clipping"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def validate_epoch(self, val_loader, criterion):
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
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
    
    def save_model(self, filepath):
        """Save complete model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'emotion_labels': self.emotion_labels,
            'max_length': self.max_length,
            'model_config': {
                'vocab_size': len(self.vocab),
                'embedding_dim': 200,
                'hidden_dim': 256,
                'num_layers': 2,
                'num_classes': len(self.emotion_labels),
                'dropout': 0.3
            }
        }, filepath)
        print(f"Model saved to: {filepath}")

def main():
    """Main training function"""
    print("🚀 IMPROVED LSTM Model Training - Target 85.79%")
    print("=" * 60)
    
    # Initialize detector
    detector = ImprovedLSTMDetector(max_length=256)
    
    # Prepare data
    data_path = 'data/processed_uit_vsfc_data.csv'
    train_df, val_df, test_df = detector.prepare_data(data_path)
    
    print(f"\n📊 Data distribution:")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    for emotion in train_df['emotion'].value_counts().index:
        count = train_df['emotion'].value_counts()[emotion]
        percentage = count / len(train_df) * 100
        print(f"  {emotion}: {count} ({percentage:.1f}%)")
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = detector.create_data_loaders(
        train_df, val_df, test_df, batch_size=32
    )
    
    # Create model
    detector.create_model()
    
    # Train model
    training_history = detector.train_improved(
        train_loader, val_loader, class_weights,
        epochs=30, learning_rate=0.001
    )
    
    # Evaluate model
    test_results = detector.evaluate(test_loader)
    
    # Save models
    improved_path = os.path.join(MODELS_DIR, 'improved_lstm_emotion_model.pth')
    torch.save(detector.model.state_dict(), improved_path)
    
    complete_path = os.path.join(MODELS_DIR, 'improved_lstm_complete.pth')
    detector.save_model(complete_path)
    
    # Save results
    results = {
        'model_name': 'LSTM',
        'training_history': training_history,
        'test_results': test_results
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'lstm_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*60)
    print("🎉 IMPROVED TRAINING COMPLETE!")
    print(f"📊 Best Validation Accuracy: {training_history['best_val_accuracy']:.4f}")
    print(f"📊 Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"📊 Test F1 Score: {test_results['f1_score']:.4f}")
    
    # Success evaluation
    if test_results['accuracy'] >= 0.85:
        print("🎯 TARGET ACHIEVED: 85%+ accuracy!")
    elif test_results['accuracy'] >= 0.80:
        print("📈 EXCELLENT: 80%+ accuracy!")
    elif test_results['accuracy'] >= 0.75:
        print("✅ GOOD: 75%+ accuracy!")
    else:
        print("🔄 NEEDS IMPROVEMENT - Consider PhoBERT")
    
    return training_history, test_results

if __name__ == "__main__":
    main() 