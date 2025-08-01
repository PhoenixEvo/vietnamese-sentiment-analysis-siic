"""
Configuration file for Sentiment Analysis project
"""
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Emotion categories (3-class for original UIT-VSFC dataset)
EMOTION_LABELS = {
    0: "negative",
    1: "neutral", 
    2: "positive"
}

LABEL_TO_ID = {v: k for k, v in EMOTION_LABELS.items()}

# Alternative 4-class mapping if needed
EMOTION_LABELS_4CLASS = {
    0: "angry",
    1: "sad",
    2: "neutral",
    3: "happy"
}

LABEL_TO_ID_4CLASS = {v: k for k, v in EMOTION_LABELS_4CLASS.items()}

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Text preprocessing parameters
MAX_LENGTH = 256  # Maximum sequence length for BERT
VOCAB_SIZE = 10000  # For traditional models

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5

# Vietnamese stopwords - Optimized for sentiment analysis
# IMPORTANT: Removed negation words (không, chưa, chẳng, etc.) and intensity words (rất, quá, lắm)
# as they are crucial for sentiment detection accuracy
VIETNAMESE_STOPWORDS = [
    # Vowel variants (tones)
    'à', 'á', 'ạ', 'ả', 'ã', 'â', 'ầ', 'ấ', 'ậ', 'ẩ', 'ẫ', 'ă', 'ằ', 'ắ', 'ặ', 'ẳ', 'ẵ',
    'è', 'é', 'ẹ', 'ẻ', 'ẽ', 'ê', 'ề', 'ế', 'ệ', 'ể', 'ễ',
    'ì', 'í', 'ị', 'ỉ', 'ĩ',
    'ò', 'ó', 'ọ', 'ỏ', 'õ', 'ô', 'ồ', 'ố', 'ộ', 'ổ', 'ỗ', 'ơ', 'ờ', 'ớ', 'ợ', 'ở', 'ỡ',
    'ù', 'ú', 'ụ', 'ủ', 'ũ', 'ư', 'ừ', 'ứ', 'ự', 'ử', 'ữ',
    'ỳ', 'ý', 'ỵ', 'ỷ', 'ỹ',
    'đ',
    # Pronouns and particles
    'anh', 'chị', 'em', 'tôi', 'tớ', 'mình', 'bạn', 'ạ', 'ơi', 'à', 'ấy', 'vậy', 'thế', 'này', 'kia', 'đó', 'đây',
    # Prepositions and conjunctions
    'và', 'với', 'của', 'cho', 'từ', 'trong', 'ngoài', 'trên', 'dưới', 'sau', 'trước', 'giữa',
    # Common verbs and auxiliary words (sentiment-neutral)
    'là', 'được', 'có', 'đã', 'sẽ', 'đang', 'bị', 'phải', 'cần', 'nên', 'còn', 'đều', 'cũng'
    # REMOVED: 'không', 'chưa', 'chẳng' (negation words - critical for sentiment)
    # REMOVED: 'rất', 'quá', 'lắm' (intensity words - important for sentiment strength)
]

# Model configurations
MODEL_CONFIGS = {
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    },
    'lstm': {
        'embedding_dim': 100,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3
    },
} 