"""
PhoBERT Model for Vietnamese Emotion Detection
Advanced transformer-based approach using pre-trained Vietnamese BERT
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import os
import sys
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Suppress transformers warnings
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
    MODELS_DIR, RESULTS_DIR, RANDOM_STATE
)

class VietnameseEmotionDataset(Dataset):
    """Dataset class for PhoBERT emotion classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        # Handle missing text
        if pd.isna(text) or text == "":
            text = "không có nội dung"
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class PhoBERTEmotionClassifier(nn.Module):
    """PhoBERT-based emotion classifier"""
    
    def __init__(self, model_name='vinai/phobert-base', num_classes=3, dropout=0.3):
        super(PhoBERTEmotionClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained PhoBERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name)
        
        # Freeze early layers (optional - can be removed for full fine-tuning)
        for param in self.phobert.embeddings.parameters():
            param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids, attention_mask):
        # Get PhoBERT outputs
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class PhoBERTEmotionDetector:
    """Main class for PhoBERT-based emotion detection"""
    
    def __init__(self, model_name='vinai/phobert-base', max_length=256):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Model will be initialized in create_model()
        self.model = None
        
        # Label encoding
        self.label_encoder = LABEL_TO_ID
        self.emotion_labels = EMOTION_LABELS
        self.id_to_label = {v: k for k, v in LABEL_TO_ID.items()}
        

        
    def prepare_data(self, data_path=None):
        """Prepare data for training"""
        if data_path is None:
            # Use UIT-VSFC data (larger dataset) for training
            uit_path = 'data/processed_uit_vsfc_data.csv'
            sample_path = os.path.join(PROCESSED_DATA_DIR, "processed_sample_data.csv")
            
            if os.path.exists(uit_path):
                data_path = uit_path
                print("Using UIT-VSFC data (large dataset)")
            else:
                data_path = sample_path
                print("Using processed sample data (fallback)")
        
        print(f"Loading data from: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        
        # Handle different column names
        if 'emotion' in df.columns and 'text' in df.columns:
            # Standard format: text, emotion, processed_text
            text_col = 'text'
            label_col = 'emotion'
        elif 'label' in df.columns:
            # Alternative format
            text_col = 'text'
            label_col = 'label'
        else:
            raise ValueError(f"Cannot find appropriate text and label columns in {list(df.columns)}")
        
        # Handle missing values
        df = df.dropna(subset=[text_col, label_col])
        print(f"After removing missing values: {len(df)} records")
        
        # Map labels to integers
        df['label_id'] = df[label_col].map(self.label_encoder)
        
        # Remove any unmapped labels
        df = df.dropna(subset=['label_id'])
        df['label_id'] = df['label_id'].astype(int)
        
        # Rename columns for consistency
        df = df.rename(columns={text_col: 'text', label_col: 'label'})
        
        print("Label distribution:")
        print(df['label'].value_counts())
        
        return df
    
    def create_data_loaders(self, train_df, val_df, test_df, batch_size=16):
        """Create data loaders for training"""
        print(f"Creating data loaders with batch size: {batch_size}")
        
        # Create datasets
        train_dataset = VietnameseEmotionDataset(
            train_df['text'], train_df['label_id'], 
            self.tokenizer, self.max_length
        )
        
        val_dataset = VietnameseEmotionDataset(
            val_df['text'], val_df['label_id'], 
            self.tokenizer, self.max_length
        )
        
        test_dataset = VietnameseEmotionDataset(
            test_df['text'], test_df['label_id'], 
            self.tokenizer, self.max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def create_model(self):
        """Create and initialize PhoBERT model"""
        print("Creating PhoBERT model...")
        
        self.model = PhoBERTEmotionClassifier(
            model_name=self.model_name,
            num_classes=len(self.emotion_labels),
            dropout=0.3
        )
        
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def train_epoch(self, model, train_loader, optimizer, scheduler, criterion):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
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
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=5, learning_rate=2e-5):
        """Train the model"""
        print(f"Starting training for {epochs} epochs...")
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_accuracy = 0
        best_model_state = None
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(
                self.model, train_loader, optimizer, scheduler, criterion
            )
            
            # Validate
            val_loss, val_acc = self.validate_epoch(self.model, val_loader, criterion)
            
            # Record history
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"✅ New best validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                print(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print("🛑 Early stopping triggered")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"✅ Loaded best model with validation accuracy: {best_val_accuracy:.4f}")
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_accuracy
        }
    
    def evaluate(self, test_loader):
        """Evaluate the model on test set"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        print("Evaluating model on test set...")
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=-1)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"\n📊 Test Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Classification report
        target_names = [self.id_to_label[i] for i in range(len(self.emotion_labels))]
        report = classification_report(all_labels, all_predictions, target_names=target_names)
        print(f"\n📋 Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print(f"\n🔄 Confusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def predict_emotion(self, text):
        """Predict emotion for a single text"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        self.model.eval()
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
        
        # Get results
        predicted_id = prediction.item()
        predicted_emotion = self.id_to_label[predicted_id]
        confidence = probabilities[0][predicted_id].item()
        
        # Get all probabilities - use proper emotion mapping from config
        prob_dict = {}
        for i in range(len(EMOTION_LABELS)):
            emotion_name = EMOTION_LABELS[i]  # 0: negative, 1: neutral, 2: positive
            prob_dict[emotion_name] = probabilities[0][i].item()
        
        return {
            'emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': prob_dict
        }
    
    def save_model(self, filepath=None):
        """Save the model"""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "phobert_emotion_model.pth")
        
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model state dict and metadata
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'label_encoder': self.label_encoder,
            'emotion_labels': self.emotion_labels
        }
        
        torch.save(save_dict, filepath)
        print(f"✅ Model saved to: {filepath}")
        
        # Also save tokenizer
        tokenizer_dir = os.path.join(MODELS_DIR, "phobert_tokenizer")
        self.tokenizer.save_pretrained(tokenizer_dir)
        print(f"✅ Tokenizer saved to: {tokenizer_dir}")
    
    def load_model(self, filepath=None):
        """Load a saved model"""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "phobert_emotion_model.pth")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        

        
        # Load model data
        save_dict = torch.load(filepath, map_location=self.device)
        
        # Update attributes
        self.model_name = save_dict['model_name']
        self.max_length = save_dict['max_length']
        self.label_encoder = save_dict['label_encoder']
        self.emotion_labels = save_dict['emotion_labels']
        self.id_to_label = {v: k for k, v in self.label_encoder.items()}
        

        
        # Create and load model
        self.model = PhoBERTEmotionClassifier(
            model_name=self.model_name,
            num_classes=len(self.emotion_labels)
        )
        
        # Load state dict with strict=False to handle position_ids issue
        try:
            self.model.load_state_dict(save_dict['model_state_dict'], strict=True)
        except RuntimeError as e:
            if "position_ids" in str(e):
                print("⚠️ Handling position_ids issue, loading with strict=False")
                # Filter out position_ids from state_dict
                state_dict = save_dict['model_state_dict']
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                     if 'position_ids' not in k}
                self.model.load_state_dict(filtered_state_dict, strict=False)
            else:
                raise e
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        tokenizer_dir = os.path.join(MODELS_DIR, "phobert_tokenizer")
        if os.path.exists(tokenizer_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        else:
            print("⚠️ Tokenizer directory not found, using default")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        


def main():
    """Main training function"""
    print("Starting PhoBERT training...")
    
    # Initialize detector
    detector = PhoBERTEmotionDetector()
    
    # Prepare data
    df = detector.prepare_data()
    
    # Split data (70-15-15)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=RANDOM_STATE, stratify=df['label_id']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df['label_id']
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create data loaders with smaller batch size for memory efficiency
    train_loader, val_loader, test_loader = detector.create_data_loaders(
        train_df, val_df, test_df, batch_size=8
    )
    
    # Train model
    history = detector.train(train_loader, val_loader, epochs=5)
    
    # Evaluate
    results = detector.evaluate(test_loader)
    
    # Save model
    detector.save_model()
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, "phobert_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump({**results, 'training_history': history}, f)
    
    print(f"✅ Results saved to: {results_path}")
    print("🎉 PhoBERT training completed!")

if __name__ == "__main__":
    main() 