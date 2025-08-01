"""
Baseline models for emotion classification
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config.config import (
    EMOTION_LABELS, LABEL_TO_ID, PROCESSED_DATA_DIR, 
    MODELS_DIR, RESULTS_DIR, RANDOM_STATE, TEST_SIZE
)

class BaselineModels:
    """Baseline models for emotion classification"""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.label_encoder = LABEL_TO_ID
        
    def prepare_data(self, data_path: str):
        """Prepare data for training"""
        # Load processed data
        df = pd.read_csv(data_path)
        
        # Filter out rows with NaN processed_text
        df = df.dropna(subset=['processed_text'])
        print(f"After removing NaN: {len(df)} records")
        
        # Encode labels
        df['label_encoded'] = df['emotion'].map(self.label_encoder)
        
        # Use the same split as LSTM model for fair comparison
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
        
        X_train = train_df['processed_text']
        y_train = train_df['label_encoded']
        X_test = test_df['processed_text']
        y_test = test_df['label_encoded']
        
        print(f"Train size: {len(X_train)}")
        print(f"Test size: {len(X_test)}")
        print(f"Label distribution in train: {train_df['emotion'].value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def create_features(self, X_train, X_test):
        """Create TF-IDF features"""
        print("Creating TF-IDF features...")
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None,  # We already removed stopwords in preprocessing
            min_df=2,
            max_df=0.95
        )
        
        # Fit and transform training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        return X_train_tfidf, X_test_tfidf
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        print("Training Logistic Regression...")
        
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        return model
    
    def train_svm(self, X_train, y_train):
        """Train SVM model"""
        print("Training SVM...")
        
        model = SVC(
            kernel='rbf',
            random_state=RANDOM_STATE,
            class_weight='balanced',
            probability=True  # Enable probability estimates
        )
        
        model.fit(X_train, y_train)
        self.models['svm'] = model
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=list(EMOTION_LABELS.values()),
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {report['accuracy']:.4f}")
        
        return {
            'model_name': model_name,
            'f1_score': f1,
            'accuracy': report['accuracy'],
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
    
    def save_models(self):
        """Save trained models and vectorizer"""
        # Ensure models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Save vectorizer
        vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Vectorizer saved to: {vectorizer_path}")
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
            joblib.dump(model, model_path)
            print(f"{model_name} saved to: {model_path}")
    
    def load_models(self):
        """Load trained models and vectorizer"""
        # Load vectorizer
        vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
            print("Vectorizer loaded successfully")
        
        # Load models
        model_files = {
            'logistic_regression': 'logistic_regression.pkl',
            'random_forest': 'random_forest.pkl', 
            'svm': 'svm.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"{model_name} loaded successfully")
    
    def predict_emotion(self, text: str, model_name: str = 'logistic_regression'):
        """Predict emotion for a single text"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if self.vectorizer is None:
            raise ValueError("Vectorizer not loaded")
        
        # Basic text preprocessing
        import re
        processed_text = text.lower().strip()
        # Remove extra whitespace
        processed_text = re.sub(r'\s+', ' ', processed_text)
        
        # Transform text using fitted vectorizer
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Predict
        model = self.models[model_name]
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]
        
        # Map prediction to emotion label
        if prediction in EMOTION_LABELS:
            predicted_emotion = EMOTION_LABELS[prediction]
        else:
            # Fallback for unexpected predictions
            predicted_emotion = "neutral"
        
        # Create probabilities dict
        emotion_probs = {}
        for i, prob in enumerate(probabilities):
            if i in EMOTION_LABELS:
                emotion_probs[EMOTION_LABELS[i]] = float(prob)
        
        return {
            'emotion': predicted_emotion,
            'confidence': float(max(probabilities)),
            'probabilities': emotion_probs
        }

def main():
    """Main training function"""
    # Ensure directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize baseline models
    baseline = BaselineModels()
    
    # Find available processed data
    uit_vsfc_path = os.path.join(PROCESSED_DATA_DIR, 'processed_uit_vsfc_data.csv')
    sample_path = os.path.join(PROCESSED_DATA_DIR, 'processed_sample_data.csv')
    
    if os.path.exists(uit_vsfc_path):
        data_path = uit_vsfc_path
        data_type = "UIT-VSFC"
        print(f"📁 Using UIT-VSFC dataset: {data_path}")
    elif os.path.exists(sample_path):
        data_path = sample_path
        data_type = "Sample"
        print(f"📝 Using sample dataset: {data_path}")
    else:
        print("❌ No processed data found!")
        print("Please run preprocessing first: python src/data_processing/preprocess.py")
        return
    
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = baseline.prepare_data(data_path)
    
    # Create features
    X_train_tfidf, X_test_tfidf = baseline.create_features(X_train, X_test)
    
    # Train models
    models_to_train = [
        ('logistic_regression', baseline.train_logistic_regression),
        ('random_forest', baseline.train_random_forest),
        ('svm', baseline.train_svm)
    ]
    
    results = []
    
    for model_name, train_func in models_to_train:
        try:
            # Train model
            model = train_func(X_train_tfidf, y_train)
            
            # Evaluate model
            result = baseline.evaluate_model(model, X_test_tfidf, y_test, model_name)
            results.append(result)
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    # Save models
    baseline.save_models()
    
    # Add data type to results
    for result in results:
        result['data_type'] = data_type
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(RESULTS_DIR, f'baseline_results_{data_type.lower()}.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {results_path}")
    
    # Display results summary
    print("\n" + "="*50)
    print(f"BASELINE MODELS RESULTS SUMMARY ({data_type} Dataset)")
    print("="*50)
    for result in results:
        print(f"{result['model_name']}: F1={result['f1_score']:.4f}, Accuracy={result['accuracy']:.4f}")

if __name__ == "__main__":
    main() 