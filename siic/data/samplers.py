"""
Create large sample data that mimics UIT-VSFC dataset structure
"""
import pandas as pd
import numpy as np
import random
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from siic.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def create_large_vietnamese_sample():
    """Create large sample Vietnamese feedback data similar to UIT-VSFC"""
    
    # Templates for different emotions
    templates = {
        'happy': [
            "Thầy/Cô {} rất tận tâm và nhiệt tình",
            "Môn học {} rất hay và bổ ích", 
            "Tôi rất hài lòng với {}",
            "Phương pháp giảng dạy {} tuyệt vời",
            "Cảm thấy vui vẻ khi học {}",
            "Được học nhiều kiến thức từ {}",
            "Thầy/Cô {} giảng bài rất dễ hiểu",
            "Môn {} giúp tôi hiểu biết nhiều hơn",
            "Rất thích cách {} của thầy/cô",
            "Bài giảng {} rất sinh động và hấp dẫn"
        ],
        'sad': [
            "Cảm thấy buồn vì {} không như mong đợi",
            "Thất vọng với {}",
            "Môn học {} khó hiểu quá",
            "Không theo kịp {} được",
            "Cảm thấy nản lòng với {}",
            "Stress quá vì {}",
            "Khó khăn trong việc học {}",
            "Không hiểu bài giảng {}",
            "Cô đơn khi học {}",
            "Mệt mỏi với {} này"
        ],
        'angry': [
            "Tức giận với {} kém chất lượng",
            "Bực mình vì {} không chuyên nghiệp", 
            "Phẫn nộ với cách {} của thầy/cô",
            "Khó chịu với {} này",
            "Bức xúc vì {} không rõ ràng",
            "Điên tiết với thái độ {}",
            "Cáu gắt vì {} tệ",
            "Chán ghét {} không hiệu quả",
            "Nổi giận với {} kém",
            "Bực bội vì {} thiếu chuyên nghiệp"
        ],
        'neutral': [
            "Môn {} bình thường thôi",
            "Cũng được, {} tạm ổn",
            "Không có ý kiến gì về {}",
            "Như vậy cũng được với {}",
            "Bình thường, {} không có gì đặc biệt",
            "Ổn, {} cũng tạm được",
            "Không tệ lắm với {}",
            "Trung bình với {}",
            "Bình thường như mọi khi với {}",
            "Cũng được thôi với {}"
        ]
    }
    
    # Subjects and topics for variation
    subjects = [
        "thầy Nguyễn", "cô Trần", "giảng viên", "thầy/cô", "môn Toán",
        "môn Lý", "môn Hóa", "môn Văn", "môn Anh", "bài giảng", 
        "phòng thực hành", "thiết bị", "tài liệu", "phương pháp giảng dạy",
        "cách trình bày", "nội dung bài học", "bài tập", "đề thi",
        "chương trình học", "thời gian học"
    ]
    
    # Generate samples
    all_samples = []
    target_samples = 1000  # Create 1000 samples
    
    for emotion, template_list in templates.items():
        samples_per_emotion = target_samples // 4  # 250 per emotion
        
        for i in range(samples_per_emotion):
            # Random template and subject
            template = random.choice(template_list)
            subject = random.choice(subjects)
            
            # Generate text
            text = template.format(subject)
            
            # Add some variation
            if random.random() < 0.3:  # 30% chance to add prefix
                prefixes = ["Thực sự", "Cá nhân tôi nghĩ", "Theo ý kiến của tôi", "Nhìn chung"]
                text = f"{random.choice(prefixes)}, {text.lower()}"
            
            if random.random() < 0.2:  # 20% chance to add suffix  
                suffixes = ["ạ", "!", ".", " lắm", " quá"]
                text = text + random.choice(suffixes)
            
            # Convert emotion to sentiment (like UIT-VSFC)
            if emotion == 'happy':
                sentiment = 2  # positive
            elif emotion == 'neutral':
                sentiment = 1  # neutral  
            else:  # sad or angry
                sentiment = 0  # negative
            
            # Assign split
            rand = random.random()
            if rand < 0.7:
                split = 'train'
            elif rand < 0.85:
                split = 'validation'
            else:
                split = 'test'
            
            all_samples.append({
                'text': text,
                'sentiment': sentiment,
                'emotion': emotion,
                'split': split
            })
    
    # Shuffle samples
    random.shuffle(all_samples)
    
    return pd.DataFrame(all_samples)

def create_realistic_feedback_data():
    """Create more realistic student feedback data"""
    
    feedback_examples = [
        # Happy feedbacks
        ("Thầy giảng bài rất hay và dễ hiểu, tôi học được nhiều kiến thức bổ ích", "happy"),
        ("Cô rất tận tâm và nhiệt tình hướng dẫn, tôi cảm thấy rất hài lòng", "happy"),
        ("Môn học này thật sự thú vị và hữu ích cho chuyên ngành của tôi", "happy"),
        ("Phòng học và thiết bị rất tốt, hỗ trợ tốt cho việc học", "happy"),
        ("Thầy luôn đi dạy đúng giờ và chuẩn bị bài kỹ lưỡng", "happy"),
        
        # Sad feedbacks  
        ("Môn học khó quá, tôi không theo kịp được", "sad"),
        ("Cảm thấy stress vì áp lực học tập quá lớn", "sad"),
        ("Buồn vì không hiểu bài giảng của thầy", "sad"),
        ("Thất vọng vì kết quả học tập không như mong muốn", "sad"),
        ("Cảm thấy cô đơn khi học online", "sad"),
        
        # Angry feedbacks
        ("Thầy dạy không rõ ràng và thái độ không tốt", "angry"),
        ("Tức giận vì thiết bị trong phòng học bị hỏng", "angry"),
        ("Bực mình vì lịch học thay đổi liên tục", "angry"),
        ("Phẫn nộ với cách chấm điểm không công bằng", "angry"),
        ("Khó chịu vì phòng học quá ồn ào", "angry"),
        
        # Neutral feedbacks
        ("Môn học bình thường, không có gì đặc biệt", "neutral"),
        ("Cũng được thôi, tạm ổn", "neutral"),
        ("Không có ý kiến gì đặc biệt", "neutral"),
        ("Như vậy cũng được", "neutral"),
        ("Bình thường như mọi khi", "neutral"),
    ]
    
    # Expand with variations
    expanded_data = []
    
    for base_text, emotion in feedback_examples:
        # Add base example
        expanded_data.append((base_text, emotion))
        
        # Create variations
        for i in range(19):  # Create 19 more variations (total 20 per base)
            # Simple variations
            variations = [
                base_text + ".",
                base_text + " ạ.",
                base_text + " lắm.",
                "Theo tôi, " + base_text.lower(),
                "Cá nhân tôi nghĩ " + base_text.lower(),
                base_text + " quá.",
                "Thực sự " + base_text.lower(),
                base_text.replace("tôi", "em"),
                base_text.replace("Thầy", "Giáo viên"),
                base_text.replace("Cô", "Thầy cô"),
            ]
            
            if i < len(variations):
                expanded_data.append((variations[i], emotion))
            else:
                # Simple repetition with minor changes
                expanded_data.append((base_text, emotion))
    
    # Convert to DataFrame
    all_samples = []
    for text, emotion in expanded_data:
        # Convert emotion to sentiment
        if emotion == 'happy':
            sentiment = 2
        elif emotion == 'neutral':
            sentiment = 1
        else:
            sentiment = 0
        
        # Assign split
        rand = random.random()
        if rand < 0.7:
            split = 'train'
        elif rand < 0.85:
            split = 'validation'
        else:
            split = 'test'
        
        all_samples.append({
            'text': text,
            'sentiment': sentiment, 
            'emotion': emotion,
            'split': split
        })
    
    return pd.DataFrame(all_samples)

def save_large_sample_data(df, filename="large_sample_data"):
    """Save large sample data in UIT-VSFC format"""
    # Ensure directories exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Save main file
    main_path = os.path.join(RAW_DATA_DIR, f'{filename}.csv')
    df.to_csv(main_path, index=False)
    print(f"💾 Large sample data saved to: {main_path}")
    
    # Save splits separately
    for split in ['train', 'validation', 'test']:
        split_df = df[df['split'] == split].copy()
        split_path = os.path.join(RAW_DATA_DIR, f'{filename}_{split}.csv')
        split_df.to_csv(split_path, index=False)
        print(f"💾 {split.title()} split saved to: {split_path}")
    
    return main_path

def analyze_large_sample(df):
    """Analyze the generated large sample data"""
    print("\n" + "="*50)
    print("📊 LARGE SAMPLE DATA ANALYSIS")
    print("="*50)
    
    print(f"Total samples: {len(df):,}")
    print(f"Unique texts: {df['text'].nunique():,}")
    print(f"Average text length: {df['text'].str.len().mean():.1f} characters")
    
    print(f"\nEmotion distribution:")
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        percentage = count / len(df) * 100
        print(f"  {emotion}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nSplit distribution:")
    split_counts = df['split'].value_counts()
    for split, count in split_counts.items():
        percentage = count / len(df) * 100
        print(f"  {split}: {count:,} ({percentage:.1f}%)")

def main():
    """Main function to create large sample data"""
    print("Creating Large Sample Data (UIT-VSFC Alternative)")
    print("="*60)
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("🔄 Generating template-based samples...")
    df1 = create_large_vietnamese_sample()
    print(f"   Generated {len(df1)} template-based samples")
    
    print("🔄 Generating realistic feedback samples...")
    df2 = create_realistic_feedback_data()
    print(f"   Generated {len(df2)} realistic samples")
    
    # Combine datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Total combined samples: {len(combined_df)}")
    
    # Analyze data
    analyze_large_sample(combined_df)
    
    # Save data  
    data_path = save_large_sample_data(combined_df, "uit_vsfc_large_sample")
    
    print("\n🎉 Large sample data created successfully!")
    print(f"📁 Data saved as: {data_path}")
    print("\nThis dataset mimics UIT-VSFC structure with:")
    print("- Similar emotion categories")
    print("- Vietnamese student feedback style")
    print("- Train/validation/test splits")
    print("- Sentiment labels (0=negative, 1=neutral, 2=positive)")
    
    print("\nNext steps:")
    print("1. Run preprocessing: python src/data_processing/preprocess.py")
    print("2. Train models: python src/models/baseline_models.py")

if __name__ == "__main__":
    main() 