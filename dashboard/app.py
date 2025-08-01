"""
Streamlit Dashboard for Sentiment Analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from siic.models.baselines import BaselineModels
from siic.models.lstm import LSTMEmotionDetector
from siic.models.phobert import PhoBERTEmotionDetector
from siic.data.preprocessors import VietnameseTextPreprocessor
from siic.utils.config import EMOTION_LABELS, RESULTS_DIR, MODELS_DIR

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .emotion-positive { color: #2ca02c; }
    .emotion-negative { color: #d62728; }
    .emotion-neutral { color: #ff7f0e; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_results():
    """Load model results"""
    try:
        results_path = os.path.join(RESULTS_DIR, 'baseline_results_uit-vsfc.csv')
        if os.path.exists(results_path):
            return pd.read_csv(results_path)
        else:
            # Try alternative name
            alt_path = os.path.join(RESULTS_DIR, 'baseline_results_sample.csv')
            if os.path.exists(alt_path):
                return pd.read_csv(alt_path)
            return None
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    
    # Load PhoBERT model (Best performing)
    try:
        phobert_detector = PhoBERTEmotionDetector()
        phobert_model_path = os.path.join(MODELS_DIR, 'phobert_emotion_model.pth')
        if os.path.exists(phobert_model_path):
            phobert_detector.load_model(phobert_model_path)
            models['phobert'] = phobert_detector
            print(" PhoBERT model loaded successfully")
    except Exception as e:
        print(f" Could not load PhoBERT model: {e}")
    
    # Load baseline models
    try:
        baseline = BaselineModels()
        baseline.load_models()
        if 'logistic_regression' in baseline.models and baseline.vectorizer is not None:
            models['baseline'] = baseline
            print(" Baseline models loaded successfully")
    except Exception as e:
        print(f" Could not load baseline models: {e}")
    
    # Load Optimized LSTM model
    try:
        lstm_detector = LSTMEmotionDetector()
        # Try different possible LSTM model names (priority: complete -> improved -> optimized)
        lstm_model_paths = [
            os.path.join(MODELS_DIR, 'improved_lstm_complete.pth'),
            os.path.join(MODELS_DIR, 'improved_lstm_emotion_model.pth'),
            os.path.join(MODELS_DIR, 'optimized_lstm_emotion_model.pth')
        ]
        
        lstm_loaded = False
        for lstm_model_path in lstm_model_paths:
            if os.path.exists(lstm_model_path):
                lstm_detector.load_model(lstm_model_path)
                models['lstm'] = lstm_detector
        
                lstm_loaded = True
                break
        
        if not lstm_loaded:
            pass
            
    except Exception as e:
        print(f" Could not load LSTM model: {e}")
    
    return models if models else None

@st.cache_resource
def load_preprocessor():
    """Load text preprocessor"""
    return VietnameseTextPreprocessor()

def create_emotion_distribution_chart(predictions):
    """Create emotion distribution pie chart"""
    emotion_counts = pd.Series(predictions).value_counts()
    
    colors = {
        'positive': '#2ca02c',
        'negative': '#d62728',
        'neutral': '#ff7f0e'
    }
    
    fig = px.pie(
        values=emotion_counts.values,
        names=emotion_counts.index,
        title="Phân bố Cảm xúc",
        color=emotion_counts.index,
        color_discrete_map=colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def create_model_comparison_chart(results_df):
    """Create model comparison chart"""
    if results_df is None:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            name='F1 Score',
            x=results_df['model_name'],
            y=results_df['f1_score'],
            yaxis='y',
            offsetgroup=1,
            marker_color='skyblue'
        ),
        go.Bar(
            name='Accuracy',
            x=results_df['model_name'],
            y=results_df['accuracy'],
            yaxis='y2',
            offsetgroup=2,
            marker_color='lightcoral'
        )
    ])
    
    fig.update_layout(
        title='So sánh Hiệu suất các Model',
        xaxis_title='Model',
        yaxis=dict(title='F1 Score', side='left'),
        yaxis2=dict(title='Accuracy', side='right', overlaying='y'),
        barmode='group',
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">Phân tích cảm xúc trong các bình luận trên mạng xã hội tiếng Việt sử dụng xử lý ngôn ngữ tự nhiên</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Cài đặt")
    
    # Load models and preprocessor
    models = load_models()
    preprocessor = load_preprocessor()
    results_df = load_results()
    
    if models is None:
        st.error("Models chưa được train hoặc không load được!")
        st.info("Vui lòng chạy: `python src/models/baseline_models.py` hoặc `python optimize_lstm_model.py`")
        return
    
    # Sidebar model selection
    available_models = []
    model_options = {}
    
    # Add PhoBERT model (Priority - Best performance)
    if 'phobert' in models:
        display_name = "PhoBERT: Transformer (93.74%)"
        available_models.append(display_name)
        model_options[display_name] = ('phobert', 'phobert')
    
    # Add Optimized LSTM model
    if 'lstm' in models:
        display_name = "Optimized LSTM: BiLSTM + Attention (85.02%)"
        available_models.append(display_name)
        model_options[display_name] = ('lstm', 'lstm')
    
    # Add baseline models with updated performance metrics
    if 'baseline' in models:
        baseline_performance = {
            'svm': '85.61%',
            'random_forest': '82.90%', 
            'logistic_regression': '82.23%'
        }
        
        for model_name in models['baseline'].models.keys():
            perf = baseline_performance.get(model_name, 'N/A')
            display_name = f" {model_name.replace('_', ' ').title()} ({perf})"
            available_models.append(display_name)
            model_options[display_name] = ('baseline', model_name)
    
    if not available_models:
        st.error("Không có model nào available!")
        return
        
    selected_model_display = st.sidebar.selectbox(
        "Chọn Model:",
        available_models,
        index=0  # PhoBERT will be first
    )
    
    model_type, model_name = model_options[selected_model_display]
    
    # Model info in sidebar
    st.sidebar.write(f"**Model Type:** {model_type.title()}")
    st.sidebar.write(f"**Model Name:** {model_name}")
    
    if results_df is not None:
        # For baseline models, look up by model_name
        if model_type == 'baseline':
            model_result = results_df[results_df['model_name'] == model_name]
        else:
            # For Optimized LSTM, look for Optimized_LSTM results
            model_result = results_df[results_df['model_name'] == 'Optimized_LSTM']
            
        if not model_result.empty:
            st.sidebar.metric(
                "F1 Score",
                f"{model_result.iloc[0]['f1_score']:.4f}"
            )
            st.sidebar.metric(
                "Accuracy", 
                f"{model_result.iloc[0]['accuracy']:.4f}"
            )
    
    # Model details based on type
    if model_type == 'phobert':
        st.sidebar.info("PhoBERT: Vietnamese transformer model với 135M parameters, đạt 93.74% accuracy, 93.44% F1-score - BEST PERFORMANCE!")
    elif model_type == 'lstm':
        st.sidebar.info("Optimized BiLSTM + Attention model với 4.5M parameters, đạt 85.02% accuracy, 85.56% F1-score trên UIT-VSFC dataset")
    else:
        if model_name == 'svm':
            st.sidebar.info(" Support Vector Machine model với TF-IDF features, đạt 85.61% accuracy, 84.88% F1-score")
        elif model_name == 'random_forest':
            st.sidebar.info(" Random Forest model với TF-IDF features, đạt 82.90% accuracy, 81.93% F1-score")
        elif model_name == 'logistic_regression':
            st.sidebar.info(" Logistic Regression model với TF-IDF features, đạt 82.23% accuracy, 83.42% F1-score")
        else:
            st.sidebar.info(" Traditional machine learning model với TF-IDF features")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Phân tích Cảm xúc", "Phân tích Batch", "Kết quả Model", "Thông tin Project"])
    
    with tab1:
        st.header("Phân tích Cảm xúc cho Text")
        
        # Text input
        user_input = st.text_area(
            "Nhập bình luận tiếng Việt:",
            placeholder="Ví dụ: Hôm nay tôi rất vui vì được gặp bạn bè!",
            height=100
        )
        
        if st.button("🔍 Phân tích Cảm xúc", type="primary"):
            if user_input.strip():
                try:
                    # Preprocess text for traditional models (not for PhoBERT)
                    processed_text = preprocessor.preprocess_text(user_input)
                    
                    # Predict emotion based on model type
                    if model_type == 'phobert':
                        # PhoBERT uses original text (preserves negation words)
                        result = models['phobert'].predict_emotion(user_input)
                        text_used = user_input.strip()
                    elif model_type == 'baseline':
                        # Baseline models use processed text
                        result = models['baseline'].predict_emotion(processed_text, model_name)
                        text_used = processed_text
                    else:  # LSTM
                        # LSTM uses processed text
                        result = models['lstm'].predict_emotion(processed_text)
                        text_used = processed_text
                    
                    # Display results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Kết quả Dự đoán")
                        emotion = result['emotion']
                        confidence = result['confidence']
                        
                        # Emotion emoji mapping
                        emotion_emojis = {
                            'positive': '😊',
                            'negative': '😢',
                            'neutral': '😐'
                        }
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{emotion_emojis.get(emotion, '🤔')} Cảm xúc: <span class="emotion-{emotion}">{emotion.title()}</span></h3>
                            <p><strong>Độ tin cậy:</strong> {confidence:.2%}</p>
                            <p><strong>Model:</strong> {selected_model_display}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show text used by the model
                        if model_type == 'phobert':
                            st.success(f"**Text gốc (PhoBERT):** {text_used}")
                        else:
                            st.info(f"**Text đã xử lý:** {text_used}")
                    
                    with col2:
                        st.subheader("Phân bố Xác suất")
                        prob_df = pd.DataFrame(
                            list(result['probabilities'].items()),
                            columns=['Cảm xúc', 'Xác suất']
                        )
                        
                        fig = px.bar(
                            prob_df,
                            x='Cảm xúc',
                            y='Xác suất',
                            color='Cảm xúc',
                            color_discrete_map={
                                'happy': '#2ca02c',
                                'sad': '#1f77b4',
                                'angry': '#d62728', 
                                'neutral': '#ff7f0e'
                            }
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show probability table
                        st.dataframe(prob_df, hide_index=True)
                        
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán: {e}")
            else:
                st.warning("Vui lòng nhập text để phân tích.")
    
    with tab2:
        st.header("Phân tích Batch")
        
        # Initialize session state for storing results
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = None
        if 'batch_df' not in st.session_state:
            st.session_state.batch_df = None
        if 'batch_predictions' not in st.session_state:
            st.session_state.batch_predictions = None
        if 'batch_confidences' not in st.session_state:
            st.session_state.batch_confidences = None
        if 'batch_results_df' not in st.session_state:
            st.session_state.batch_results_df = None
        if 'batch_model_used' not in st.session_state:
            st.session_state.batch_model_used = None
        if 'batch_timestamp' not in st.session_state:
            st.session_state.batch_timestamp = None
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload file CSV chứa bình luận:",
            type=['csv'],
            help="File CSV cần có cột 'text' chứa các bình luận"
        )
        
        # Check if we have previous results to show (even without new file upload)
        if st.session_state.batch_results and st.session_state.batch_predictions and st.session_state.batch_results_df is not None and uploaded_file is None:
            model_info = f" với {st.session_state.batch_model_used}" if st.session_state.batch_model_used else ""
            timestamp_info = f" - {st.session_state.batch_timestamp}" if st.session_state.batch_timestamp else ""
            st.success(f" Kết quả phân tích trước đó ({len(st.session_state.batch_predictions)} bình luận{model_info}){timestamp_info}:")
            
            # Display previous results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Kết quả Phân tích")
                st.dataframe(
                    st.session_state.batch_results_df[['text', 'emotion', 'confidence']].head(10),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Phân bố Cảm xúc")
                fig = create_emotion_distribution_chart(st.session_state.batch_predictions)
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.subheader("Thống kê Chi tiết")
            emotion_stats = pd.Series(st.session_state.batch_predictions).value_counts()
            
            col1, col2, col3 = st.columns(3)
            emotions = ['positive', 'negative', 'neutral']
            
            for i, emotion in enumerate(emotions):
                count = emotion_stats.get(emotion, 0)
                percentage = count / len(st.session_state.batch_predictions) * 100
                
                with [col1, col2, col3][i]:
                    st.metric(
                        label=f"{emotion.title()}",
                        value=count,
                        delta=f"{percentage:.1f}%"
                    )
            
            # Average confidence
            st.metric("Độ tin cậy trung bình", f"{np.mean(st.session_state.batch_confidences):.2%}")
            
            # Download section for previous results
            st.subheader("📥 Tải Kết quả")
            csv_data = st.session_state.batch_results_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Download CSV (Excel Compatible)",
                data=csv_data,
                file_name="emotion_analysis_results.csv",
                mime="text/csv",
                key="download_csv_standalone"
            )
            
            # Instructions for Excel
            st.info("💡 **Hướng dẫn mở file trong Excel:**\n"
                   "1. Mở Excel\n"
                   "2. Data → From Text/CSV\n"
                   "3. Chọn file CSV\n"
                   "4. File Origin: UTF-8\n"
                   "5. Load")
            
            st.divider()
            
            # Add button to clear previous results
            if st.button("🗑️ Xóa Kết quả Cũ", type="secondary"):
                st.session_state.batch_results = None
                st.session_state.batch_predictions = None
                st.session_state.batch_confidences = None
                st.session_state.batch_results_df = None
                st.session_state.batch_model_used = None
                st.session_state.batch_timestamp = None
                st.rerun()
        
        # Handle file upload and analysis
        if uploaded_file:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                
                if 'text' not in df.columns:
                    st.error("File CSV cần có cột 'text'")
                else:
                    st.success(f"Đã load {len(df)} bình luận.")
                    
                    # Show sample data
                    st.subheader("Preview Data")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Store original dataframe
                    st.session_state.batch_df = df.copy()
                    
                    # Clear previous results when new file is uploaded
                    st.session_state.batch_results = None
                    st.session_state.batch_predictions = None
                    st.session_state.batch_confidences = None
                    st.session_state.batch_results_df = None
                    st.session_state.batch_model_used = None
                    st.session_state.batch_timestamp = None
                    
                    # Check if we have previous results to show
                    if st.session_state.batch_results and st.session_state.batch_predictions and st.session_state.batch_results_df is not None:
                        model_info = f" với {st.session_state.batch_model_used}" if st.session_state.batch_model_used else ""
                        timestamp_info = f" - {st.session_state.batch_timestamp}" if st.session_state.batch_timestamp else ""
                        st.success(f" Kết quả phân tích trước đó ({len(st.session_state.batch_predictions)} bình luận{model_info}){timestamp_info}:")
                        
                        # Display previous results
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("Kết quả Phân tích")
                            st.dataframe(
                                st.session_state.batch_results_df[['text', 'emotion', 'confidence']].head(10),
                                use_container_width=True
                            )
                        
                        with col2:
                            st.subheader("Phân bố Cảm xúc")
                            fig = create_emotion_distribution_chart(st.session_state.batch_predictions)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        st.subheader("Thống kê Chi tiết")
                        emotion_stats = pd.Series(st.session_state.batch_predictions).value_counts()
                        
                        col1, col2, col3 = st.columns(3)
                        emotions = ['positive', 'negative', 'neutral']
                        
                        for i, emotion in enumerate(emotions):
                            count = emotion_stats.get(emotion, 0)
                            percentage = count / len(st.session_state.batch_predictions) * 100
                            
                            with [col1, col2, col3][i]:
                                st.metric(
                                    label=f"{emotion.title()}",
                                    value=count,
                                    delta=f"{percentage:.1f}%"
                                )
                        
                        # Average confidence
                        st.metric("Độ tin cậy trung bình", f"{np.mean(st.session_state.batch_confidences):.2%}")
                        
                        # Download section for previous results
                        st.subheader("📥 Tải Kết quả")
                        csv_data = st.session_state.batch_results_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Download CSV (Excel Compatible)",
                            data=csv_data,
                            file_name="emotion_analysis_results.csv",
                            mime="text/csv",
                            key="download_csv_previous"
                        )
                        
                        # Instructions for Excel
                        st.info("💡 **Hướng dẫn mở file trong Excel:**\n"
                               "1. Mở Excel\n"
                               "2. Data → From Text/CSV\n"
                               "3. Chọn file CSV\n"
                               "4. File Origin: UTF-8\n"
                               "5. Load")
                        
                        st.divider()
                        
                        # Add button to clear previous results
                        if st.button("🗑️ Xóa Kết quả Cũ", type="secondary"):
                            st.session_state.batch_results = None
                            st.session_state.batch_predictions = None
                            st.session_state.batch_confidences = None
                            st.session_state.batch_results_df = None
                            st.session_state.batch_model_used = None
                            st.session_state.batch_timestamp = None
                            st.rerun()
                    
                    if st.button("Phân tích Tất cả", type="primary"):
                        # Process all texts
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        predictions = []
                        confidences = []
                        
                        for i, text in enumerate(df['text']):
                            status_text.text(f'Đang xử lý {i+1}/{len(df)}...')
                            
                            # Preprocess text for traditional models
                            processed_text = preprocessor.preprocess_text(str(text))
                            
                            # Use appropriate text based on model type
                            if model_type == 'phobert':
                                # PhoBERT uses original text
                                if str(text).strip():
                                    result = models['phobert'].predict_emotion(str(text))
                                else:
                                    result = {'emotion': 'neutral', 'confidence': 0.25}
                            elif processed_text.strip():
                                # Other models use processed text
                                if model_type == 'baseline':
                                    result = models['baseline'].predict_emotion(processed_text, model_name)
                                else:  # LSTM
                                    result = models['lstm'].predict_emotion(processed_text)
                            else:
                                result = {'emotion': 'neutral', 'confidence': 0.25}
                            
                            # Add results to lists
                            predictions.append(result['emotion'])
                            confidences.append(result['confidence'])
                            
                            progress_bar.progress((i + 1) / len(df))
                        
                        status_text.text('Hoàn thành!')
                        
                        # Store results in session state
                        st.session_state.batch_predictions = predictions
                        st.session_state.batch_confidences = confidences
                        st.session_state.batch_results = True
                        st.session_state.batch_model_used = selected_model_display
                        st.session_state.batch_timestamp = pd.Timestamp.now().strftime("%H:%M:%S - %d/%m/%Y")
                        
                        # Show results immediately
                        st.success(f" Phân tích hoàn thành với {selected_model_display}!")
                        
                        # Add predictions to dataframe
                        df['emotion'] = predictions
                        df['confidence'] = confidences
                        
                        # Display results
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("Kết quả Phân tích")
                            st.dataframe(
                                df[['text', 'emotion', 'confidence']].head(10),
                                use_container_width=True
                            )
                        
                        with col2:
                            st.subheader("Phân bố Cảm xúc")
                            fig = create_emotion_distribution_chart(predictions)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        st.subheader("Thống kê Chi tiết")
                        emotion_stats = pd.Series(predictions).value_counts()
                        
                        col1, col2, col3 = st.columns(3)
                        emotions = ['positive', 'negative', 'neutral']
                        
                        for i, emotion in enumerate(emotions):
                            count = emotion_stats.get(emotion, 0)
                            percentage = count / len(predictions) * 100
                            
                            with [col1, col2, col3][i]:
                                st.metric(
                                    label=f"{emotion.title()}",
                                    value=count,
                                    delta=f"{percentage:.1f}%"
                                )
                        
                        # Average confidence
                        st.metric("Độ tin cậy trung bình", f"{np.mean(confidences):.2%}")
                        
                        # Store results dataframe in session state for download
                        st.session_state.batch_results_df = df.copy()
                        
                        # Success message with persistence info
                        st.success(f" Phân tích hoàn thành với {selected_model_display}! Kết quả đã được lưu và sẽ không bị mất khi tải về file.")
                        
                        # Download section
                        st.subheader("📥 Tải Kết quả")
                        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Download CSV (Excel Compatible)",
                            data=csv_data,
                            file_name="emotion_analysis_results.csv",
                            mime="text/csv",
                            key="download_csv_immediate"
                        )
                        
                        # Instructions for Excel
                        st.info("💡 **Hướng dẫn mở file trong Excel:**\n"
                               "1. Mở Excel\n"
                               "2. Data → From Text/CSV\n"
                               "3. Chọn file CSV\n"
                               "4. File Origin: UTF-8\n"
                               "5. Load")
                        
            except Exception as e:
                st.error(f"Lỗi khi xử lý file: {e}")
        

    
    with tab3:
        st.header("Kết quả Training Model")
        
        if results_df is not None:
            # Model comparison chart
            fig = create_model_comparison_chart(results_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("Bảng Kết quả Chi tiết")
            display_df = results_df[['model_name', 'f1_score', 'accuracy']].copy()
            display_df.columns = ['Model', 'F1 Score', 'Accuracy']
            display_df['F1 Score'] = display_df['F1 Score'].round(4)
            display_df['Accuracy'] = display_df['Accuracy'].round(4)
            st.dataframe(display_df, use_container_width=True)
            
            # Best model highlight
            best_model = results_df.loc[results_df['f1_score'].idxmax()]
            st.success(f"🏆 **Best Model**: {best_model['model_name']} (F1: {best_model['f1_score']:.4f})")
            
        else:
            st.warning("Chưa có kết quả training.")
    
    with tab4:
        st.header("Thông tin Project")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('''
**Mục tiêu:** Xây dựng hệ thống phân tích sentiment (tích cực, tiêu cực, trung tính) cho bình luận mạng xã hội tiếng Việt sử dụng các kỹ thuật NLP hiện đại (PhoBERT, LSTM, SVM, v.v). Hệ thống hướng đến độ chính xác cao, tốc độ xử lý nhanh và dễ dàng mở rộng cho các ứng dụng thực tế.
''')

            st.markdown('''
### 👥 Team InsideOut
- **Team Leader**: Nguyễn Nhật Phát  
- **Thành viên**: Nguyễn Tiến Huy
''')
        
        with col2:
            # Project stats
            if models and results_df is not None:
                st.metric("Models Trained", len(results_df))
                st.metric("Best F1 Score", f"{results_df['f1_score'].max():.4f}")
                st.metric("Avg Accuracy", f"{results_df['accuracy'].mean():.4f}")
            
            # Dataset info
            try:
                data_path = os.path.join(project_root, 'data', 'processed_uit_vsfc_data.csv')
                if os.path.exists(data_path):
                    df_info = pd.read_csv(data_path)
                    st.metric("Dataset Size", f"{len(df_info):,}")
                    st.metric("Emotions", len(df_info['emotion'].unique()))
            except:
                pass
        
        # Technical stack
        st.subheader("🛠️ Technical Stack")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **NLP Libraries**
            - underthesea
            - pyvi  
            - transformers (Hugging Face)
            - nltk
            - PhoBERT tokenizer
            """)
        
        with col2:
            st.markdown("""
            **ML/DL Frameworks**
            - PyTorch (PhoBERT, LSTM)
            - scikit-learn (Baseline models)
            - transformers
            - numpy, pandas
            """)
        
        with col3:
            st.markdown("""
            **Visualization & UI**
            - Streamlit
            - Plotly
            - Matplotlib
            - Seaborn
            """)

if __name__ == "__main__":
    main() 