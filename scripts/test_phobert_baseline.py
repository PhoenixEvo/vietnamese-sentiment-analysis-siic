#!/usr/bin/env python3
"""
Test PhoBERT Baseline Accuracy (Zero-shot Performance)
Kiểm tra accuracy của PhoBERT trước khi fine-tune
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config.config import (EMOTION_LABELS, LABEL_TO_ID, MODELS_DIR,
                           PROCESSED_DATA_DIR, RANDOM_STATE, RESULTS_DIR)


class PhoBERTBaselineTester:
    """Test PhoBERT baseline performance without fine-tuning"""

    def __init__(self, model_name="vinai/phobert-base", max_length=256):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer and model
        print(f"Loading PhoBERT: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Label mapping
        self.label_encoder = LABEL_TO_ID
        self.emotion_labels = EMOTION_LABELS
        self.id_to_label = {v: k for k, v in LABEL_TO_ID.items()}

        print(f" PhoBERT loaded successfully")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def get_text_embeddings(self, texts):
        """Extract [CLS] token embeddings from PhoBERT"""
        embeddings = []

        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting embeddings"):
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                # Move to device
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                # Get model output
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                )

                # Use [CLS] token representation
                cls_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
                embeddings.append(cls_embedding[0])

        return np.array(embeddings)

    def simple_classifier(self, embeddings, labels):
        """Simple classifier using cosine similarity with class centroids"""
        from sklearn.metrics.pairwise import cosine_similarity

        # Calculate class centroids
        unique_labels = np.unique(labels)
        centroids = {}

        for label in unique_labels:
            label_embeddings = embeddings[labels == label]
            centroids[label] = np.mean(label_embeddings, axis=0)

        # Predict using cosine similarity
        predictions = []
        for embedding in embeddings:
            similarities = {}
            for label, centroid in centroids.items():
                similarity = cosine_similarity([embedding], [centroid])[0][0]
                similarities[label] = similarity

            # Predict class with highest similarity
            predicted_label = max(similarities, key=similarities.get)
            predictions.append(predicted_label)

        return np.array(predictions)

    def load_data(self):
        """Load test data"""
        # Try to load UIT-VSFC data
        uit_path = "data/processed_uit_vsfc_data.csv"
        sample_path = os.path.join(PROCESSED_DATA_DIR, "processed_sample_data.csv")

        if os.path.exists(uit_path):
            data_path = uit_path
            print(f" Using UIT-VSFC dataset: {data_path}")
        elif os.path.exists(sample_path):
            data_path = sample_path
            print(f" Using sample dataset: {data_path}")
        else:
            raise FileNotFoundError("No processed data found!")

        # Load data
        df = pd.read_csv(data_path)
        print(f" Loaded {len(df)} records")

        # Handle different column names
        if "emotion" in df.columns and "text" in df.columns:
            text_col = "text"
            label_col = "emotion"
        elif "label" in df.columns:
            text_col = "text"
            label_col = "label"
        else:
            raise ValueError(f"Cannot find appropriate columns in {list(df.columns)}")

        # Clean data
        df = df.dropna(subset=[text_col, label_col])
        df["label_id"] = df[label_col].map(self.label_encoder)
        df = df.dropna(subset=["label_id"])
        df["label_id"] = df["label_id"].astype(int)

        print(" Label distribution:")
        print(df[label_col].value_counts())

        return df[text_col].values, df["label_id"].values

    def test_baseline_performance(self, test_size=0.3):
        """Test baseline performance without fine-tuning"""
        print("\n" + "=" * 60)
        print("🧪 TESTING PHOBERT BASELINE PERFORMANCE")
        print("=" * 60)

        # Load data
        texts, labels = self.load_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=RANDOM_STATE, stratify=labels
        )

        print(f"\n Data split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")

        # Extract embeddings
        print(f"\n🔍 Extracting PhoBERT embeddings...")
        train_embeddings = self.get_text_embeddings(X_train)
        test_embeddings = self.get_text_embeddings(X_test)

        print(f"  Train embeddings shape: {train_embeddings.shape}")
        print(f"  Test embeddings shape: {test_embeddings.shape}")

        # Simple classification using centroids
        print(f"\n Simple classification using class centroids...")
        y_pred = self.simple_classifier(test_embeddings, y_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"\n BASELINE RESULTS (Zero-shot):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")

        # Classification report
        target_names = [self.id_to_label[i] for i in range(len(self.emotion_labels))]
        report = classification_report(y_test, y_pred, target_names=target_names)
        print(f"\n Classification Report:")
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n Confusion Matrix:")
        print(cm)

        # Per-class accuracy
        print(f"\n Per-class Accuracy:")
        for i, emotion in enumerate(target_names):
            class_mask = y_test == i
            if class_mask.sum() > 0:
                class_acc = (y_pred[class_mask] == y_test[class_mask]).mean()
                print(f"  {emotion}: {class_acc:.4f} ({class_mask.sum()} samples)")

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "predictions": y_pred,
            "true_labels": y_test,
            "classification_report": report,
            "confusion_matrix": cm,
        }

    def compare_with_random_baseline(self):
        """Compare with random baseline"""
        print(f"\n RANDOM BASELINE COMPARISON")

        # Load data
        texts, labels = self.load_data()

        # Random predictions
        np.random.seed(RANDOM_STATE)
        random_predictions = np.random.choice(
            np.unique(labels),
            size=len(labels),
            p=[1 / len(np.unique(labels))] * len(np.unique(labels)),
        )

        random_accuracy = accuracy_score(labels, random_predictions)
        random_f1 = f1_score(labels, random_predictions, average="weighted")

        print(f"  Random Accuracy: {random_accuracy:.4f}")
        print(f"  Random F1-Score: {random_f1:.4f}")

        return {"random_accuracy": random_accuracy, "random_f1_score": random_f1}


def main():
    """Main testing function"""
    print("🧪 PhoBERT Baseline Performance Test")
    print("=" * 50)

    # Initialize tester
    tester = PhoBERTBaselineTester()

    # Test baseline performance
    baseline_results = tester.test_baseline_performance()

    # Compare with random baseline
    random_results = tester.compare_with_random_baseline()

    # Summary
    print(f"\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"PhoBERT Baseline (Zero-shot):")
    print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"  F1-Score: {baseline_results['f1_score']:.4f}")
    print(f"\nRandom Baseline:")
    print(f"  Accuracy: {random_results['random_accuracy']:.4f}")
    print(f"  F1-Score: {random_results['random_f1_score']:.4f}")
    print(f"\nImprovement over random:")
    print(
        f"  Accuracy: +{(baseline_results['accuracy'] - random_results['random_accuracy'])*100:.1f}%"
    )
    print(
        f"  F1-Score: +{(baseline_results['f1_score'] - random_results['random_f1_score'])*100:.1f}%"
    )

    # Save results
    results_path = os.path.join(RESULTS_DIR, "phobert_baseline_results.pkl")
    import pickle

    with open(results_path, "wb") as f:
        pickle.dump({"baseline_results": baseline_results, "random_results": random_results}, f)

    print(f"\n Results saved to: {results_path}")

    # Conclusion
    print(f"\n CONCLUSION:")
    if baseline_results["accuracy"] > 0.5:
        print(f"   PhoBERT có baseline accuracy tốt ({baseline_results['accuracy']:.4f})")
        print(f"   Fine-tuning có thể cải thiện thêm")
    else:
        print(f"   PhoBERT baseline accuracy thấp ({baseline_results['accuracy']:.4f})")
        print(f"   Cần fine-tuning để cải thiện performance")


if __name__ == "__main__":
    main()
