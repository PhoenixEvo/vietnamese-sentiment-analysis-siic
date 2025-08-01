"""
Script to download and load UIT-VSFC dataset from Hugging Face
"""
import pandas as pd
import numpy as np
from datasets import load_dataset
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, EMOTION_LABELS, LABEL_TO_ID

def download_uit_vsfc():
    """Download UIT-VSFC dataset from Hugging Face"""
    print("🔄 Loading UIT-VSFC dataset from Hugging Face...")
    
    try:
        # Load dataset from Hugging Face (will use cache if available)
        dataset = load_dataset("uitnlp/vietnamese_students_feedback")
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Train: {len(dataset['train'])} samples")
        print(f"   Validation: {len(dataset['validation'])} samples") 
        print(f"   Test: {len(dataset['test'])} samples")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def convert_sentiment_to_emotion(sentiment_label):
    """Convert UIT-VSFC sentiment labels to emotion categories"""
    # UIT-VSFC: 0=negative, 1=neutral, 2=positive
    # Our project: 0=neutral, 1=happy, 2=sad, 3=angry
    
    if sentiment_label == 2:  # positive
        return 'happy'
    elif sentiment_label == 1:  # neutral
        return 'neutral'
    elif sentiment_label == 0:  # negative
        # For now, map all negative to 'sad'
        # Later can use text analysis to distinguish sad vs angry
        return 'sad'
    else:
        return 'neutral'

def advanced_emotion_mapping(text, sentiment_label):
    """Advanced mapping using keyword analysis for negative sentiments"""
    if sentiment_label != 0:  # Only process negative sentiments
        return convert_sentiment_to_emotion(sentiment_label)
    
    # Keywords for angry vs sad
    angry_keywords = [
        'tức giận', 'bực mình', 'khó chịu', 'phẫn nộ', 'điên tiết', 
        'bức xúc', 'tức tối', 'cáu gắt', 'nổi giận', 'chán ghét',
        'kém chất lượng', 'tệ', 'tồi tệ', 'không chuyên nghiệp'
    ]
    
    sad_keywords = [
        'buồn', 'thất vọng', 'chán nản', 'lo lắng', 'stress', 
        'mệt mỏi', 'cô đơn', 'không vui', 'khó khăn', 'thiệt thòi'
    ]
    
    text_lower = text.lower()
    
    # Count occurrences
    angry_count = sum(1 for keyword in angry_keywords if keyword in text_lower)
    sad_count = sum(1 for keyword in sad_keywords if keyword in text_lower)
    
    if angry_count > sad_count:
        return 'angry'
    else:
        return 'sad'

def process_uit_vsfc_dataset(dataset, use_advanced_mapping=True):
    """Process UIT-VSFC dataset and convert to our format"""
    print("🔄 Processing UIT-VSFC dataset...")
    
    # Combine all splits
    all_data = []
    
    for split_name, split_data in dataset.items():
        print(f"   Processing {split_name}: {len(split_data)} samples")
        
        for example in split_data:
            text = example['sentence']
            sentiment = example['sentiment']
            
            # Convert sentiment to emotion
            if use_advanced_mapping:
                emotion = advanced_emotion_mapping(text, sentiment)
            else:
                emotion = convert_sentiment_to_emotion(sentiment)
            
            all_data.append({
                'text': text,
                'sentiment': sentiment,
                'emotion': emotion,
                'split': split_name
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"✅ Processed {len(df)} total samples")
    print("\nEmotion distribution:")
    print(df['emotion'].value_counts())
    
    print("\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    
    return df

def save_processed_data(df):
    """Save processed dataset"""
    # Ensure directories exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save raw data
    raw_path = os.path.join(RAW_DATA_DIR, 'uit_vsfc_raw.csv')
    df.to_csv(raw_path, index=False)
    print(f"💾 Raw data saved to: {raw_path}")
    
    # Save train/val/test splits separately  
    for split in ['train', 'validation', 'test']:
        split_df = df[df['split'] == split].copy()
        split_path = os.path.join(RAW_DATA_DIR, f'uit_vsfc_{split}.csv')
        split_df.to_csv(split_path, index=False)
        print(f"💾 {split.title()} split saved to: {split_path}")
    
    return raw_path

def analyze_dataset_quality(df):
    """Analyze dataset quality and characteristics"""
    print("\n" + "="*50)
    print("📊 DATASET ANALYSIS")
    print("="*50)
    
    # Basic stats
    print(f"Total samples: {len(df):,}")
    print(f"Unique texts: {df['text'].nunique():,}")
    print(f"Average text length: {df['text'].str.len().mean():.1f} characters")
    
    # Text length distribution
    text_lengths = df['text'].str.len()
    print(f"Text length stats:")
    print(f"  Min: {text_lengths.min()}")
    print(f"  Max: {text_lengths.max()}")
    print(f"  Median: {text_lengths.median():.1f}")
    
    # Emotion balance
    print(f"\nEmotion balance:")
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        percentage = count / len(df) * 100
        print(f"  {emotion}: {count:,} ({percentage:.1f}%)")
    
    # Split distribution
    print(f"\nSplit distribution:")
    split_counts = df['split'].value_counts()
    for split, count in split_counts.items():
        percentage = count / len(df) * 100
        print(f"  {split}: {count:,} ({percentage:.1f}%)")

def main():
    """Main function to download and process UIT-VSFC"""
    print("UIT-VSFC Dataset Download and Processing")
    print("="*50)
    
    # Download dataset
    dataset = download_uit_vsfc()
    if dataset is None:
        print("❌ Failed to download dataset")
        return
    
    # Process dataset
    df = process_uit_vsfc_dataset(dataset, use_advanced_mapping=True)
    
    # Analyze dataset
    analyze_dataset_quality(df)
    
    # Save processed data
    data_path = save_processed_data(df)
    
    print("\n🎉 UIT-VSFC dataset ready for use!")
    print(f"📁 Data saved to: {data_path}")
    print("\nNext steps:")
    print("1. Run preprocessing: python src/data_processing/preprocess.py")
    print("2. Train models: python src/models/baseline_models.py") 
    print("3. Compare with sample data results")

if __name__ == "__main__":
    main() 