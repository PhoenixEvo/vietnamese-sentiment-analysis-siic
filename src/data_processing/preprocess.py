"""
Data preprocessing utilities for Vietnamese text
"""
import re
import pandas as pd
import numpy as np
from underthesea import word_tokenize, pos_tag
from pyvi import ViTokenizer
import string
import unicodedata
from typing import List, Optional
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config.config import VIETNAMESE_STOPWORDS, PROCESSED_DATA_DIR, RAW_DATA_DIR

class VietnameseTextPreprocessor:
    """Vietnamese text preprocessing class"""
    
    def __init__(self):
        self.stopwords = set(VIETNAMESE_STOPWORDS)
        
    def clean_text(self, text: str) -> str:
        """Clean Vietnamese text"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove phone numbers
        text = re.sub(r'\d{10,11}', ' ', text)
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Remove special characters but keep Vietnamese characters
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
        
        # Remove digits
        text = re.sub(r'\d+', ' ', text)
        
        # Remove excessive whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove Vietnamese stopwords"""
        return [token for token in tokens if token not in self.stopwords and len(token) > 1]
    
    def tokenize_vietnamese(self, text: str) -> List[str]:
        """Tokenize Vietnamese text using underthesea"""
        try:
            # Use underthesea for word segmentation
            tokens = word_tokenize(text)
            return tokens
        except:
            # Fallback to pyvi if underthesea fails
            tokens = ViTokenizer.tokenize(text).split()
            return tokens
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True) -> str:
        """Complete preprocessing pipeline"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_vietnamese(cleaned_text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Join tokens back to string
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str, label_column: str = None) -> pd.DataFrame:
        """Preprocess entire dataframe"""
        print(f"Preprocessing {len(df)} samples...")
        
        # Create copy of dataframe
        processed_df = df.copy()
        
        # Apply preprocessing to text column
        processed_df['processed_text'] = processed_df[text_column].apply(
            lambda x: self.preprocess_text(x)
        )
        
        # Remove empty texts
        processed_df = processed_df[processed_df['processed_text'].str.len() > 0]
        
        print(f"After preprocessing: {len(processed_df)} samples remain")
        
        return processed_df

def load_uit_vsfc_dataset(data_path: str) -> pd.DataFrame:
    """Load UIT-VSFC dataset"""
    try:
        # Assuming CSV format with columns: text, label
        df = pd.read_csv(data_path)
        
        # Map sentiment labels to emotion categories
        # This is a simplified mapping - you may need to adjust based on actual dataset
        sentiment_to_emotion = {
            'positive': 'happy',
            'negative': 'sad', 
            'neutral': 'neutral'
        }
        
        if 'sentiment' in df.columns:
            df['emotion'] = df['sentiment'].map(sentiment_to_emotion)
        
        return df
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def create_sample_data():
    """Create sample Vietnamese social media comments for testing"""
    sample_data = {
        'text': [
            # Happy emotions
            'Hôm nay tôi rất vui vì được gặp bạn bè!',
            'Yêu thích bộ phim này lắm, hay quá!',
            'Tuyệt vời! Đúng những gì tôi mong đợi',
            'Thích thú với món ăn này, ngon tuyệt!',
            'Hạnh phúc khi được làm việc ở đây',
            'Cảm thấy vui mừng và hào hứng',
            
            # Sad emotions  
            'Mình buồn quá, công việc không như ý',
            'Chán ghê, không có gì thú vị cả',
            'Thất vọng với kết quả này',
            'Cảm thấy cô đơn và buồn bã',
            'Khó chịu vì không đạt được mục tiêu',
            'Stress quá, mệt mỏi lắm',
            
            # Angry emotions
            'Thật tức giận với dịch vụ kém chất lượng này',
            'Điên tiết với thái độ phục vụ tệ',
            'Bức xúc với cách làm việc này',
            'Phẫn nộ vì bị lừa dối',
            'Khó chịu với sự thiếu chuyên nghiệp',
            'Tức giận vì phải chờ đợi quá lâu',
            
            # Neutral emotions
            'Cuộc sống bình thường như mọi ngày',
            'Ổn thôi, không có gì đặc biệt',
            'Sản phẩm tạm được, không tệ lắm',
            'Cũng bình thường thôi',
            'Không có ý kiến gì đặc biệt',
            'Như vậy cũng được rồi',
        ],
        'emotion': (
            ['happy'] * 6 + 
            ['sad'] * 6 + 
            ['angry'] * 6 + 
            ['neutral'] * 6
        )
    }
    
    return pd.DataFrame(sample_data)

def main():
    """Main preprocessing function"""
    # Ensure directories exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = VietnameseTextPreprocessor()
    
    # Check if UIT-VSFC data exists
    uit_vsfc_path = os.path.join(RAW_DATA_DIR, 'uit_vsfc_raw.csv')
    
    if os.path.exists(uit_vsfc_path):
        print("📁 Found UIT-VSFC dataset, using real data...")
        df = pd.read_csv(uit_vsfc_path)
        data_type = "uit_vsfc"
        print(f"Loaded {len(df)} samples from UIT-VSFC")
        print(f"Emotion distribution:\n{df['emotion'].value_counts()}")
    else:
        print("📝 UIT-VSFC not found, creating sample data...")
        df = create_sample_data()
        data_type = "sample"
        
        # Save raw sample data
        raw_data_path = os.path.join(RAW_DATA_DIR, 'sample_data.csv')
        df.to_csv(raw_data_path, index=False, encoding='utf-8-sig')
        print(f"Sample data saved to: {raw_data_path}")
    
    # Preprocess the data
    print(f"\n🔄 Preprocessing {data_type} data...")
    processed_df = preprocessor.preprocess_dataframe(df, 'text', 'emotion')
    
    # Save processed data
    processed_data_path = os.path.join(PROCESSED_DATA_DIR, f'processed_{data_type}_data.csv')
    processed_df.to_csv(processed_data_path, index=False, encoding='utf-8-sig')
    print(f"Processed data saved to: {processed_data_path}")
    
    # Display sample results
    print(f"\n📋 {data_type.title()} preprocessing results:")
    for i in range(min(3, len(processed_df))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Original: {df.iloc[i]['text']}")
        print(f"Processed: {processed_df.iloc[i]['processed_text']}")
        print(f"Emotion: {processed_df.iloc[i]['emotion']}")
    
    print(f"\n✅ Preprocessing completed for {len(processed_df)} samples!")

if __name__ == "__main__":
    main() 