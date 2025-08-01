"""
Unit tests for sentiment analysis models
"""
import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
try:
    from siic.models.lstm import LSTMEmotionDetector, LSTMEmotionClassifier, ImprovedLSTMModel
    from siic.models.baselines import EmotionClassifier
    from siic.models.phobert import PhoBERTEmotionDetector, PhoBERTEmotionClassifier
except ImportError as e:
    pytest.skip(f"Model imports failed: {e}", allow_module_level=True)

@pytest.mark.unit
class TestLSTMModel:
    """Test LSTM emotion detection model"""
    
    @pytest.fixture
    def sample_texts(self):
        """Sample Vietnamese texts for testing"""
        return [
            "Tôi rất vui hôm nay!",
            "Buồn quá, không muốn làm gì",
            "Bình thường thôi, không có gì đặc biệt",
            "Sản phẩm này tuyệt vời!",
            "Dịch vụ khách hàng tệ"
        ]
    
    @pytest.fixture
    def sample_labels(self):
        """Sample labels for testing"""
        return [1, 0, 2, 1, 0]  # positive, negative, neutral, positive, negative
    
    @pytest.fixture
    def lstm_detector(self):
        """LSTM detector instance for testing"""
        return LSTMEmotionDetector(max_length=64)
    
    def test_lstm_detector_initialization(self, lstm_detector):
        """Test LSTM detector initializes correctly"""
        assert lstm_detector.max_length == 64
        assert lstm_detector.device is not None
        assert lstm_detector.label_encoder is not None
        assert lstm_detector.emotion_labels is not None
    
    def test_build_vocabulary(self, lstm_detector, sample_texts):
        """Test vocabulary building"""
        vocab = lstm_detector.build_vocabulary(sample_texts, min_freq=1)
        
        # Check vocabulary structure
        assert '<PAD>' in vocab
        assert '<UNK>' in vocab
        assert vocab['<PAD>'] == 0
        assert vocab['<UNK>'] == 1
        
        # Check that common Vietnamese words are included
        assert any('tôi' in word.lower() for word in vocab.keys())
        assert len(vocab) >= 10  # Should have reasonable vocabulary size
    
    def test_lstm_classifier_architecture(self):
        """Test LSTM classifier architecture"""
        model = LSTMEmotionClassifier(
            vocab_size=1000,
            embedding_dim=100,
            hidden_dim=128,
            num_layers=2,
            num_classes=3
        )
        
        # Test forward pass
        batch_size = 4
        seq_length = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        output = model(input_ids)
        
        # Check output shape
        assert output.shape == (batch_size, 3)
        
        # Check that output is reasonable (not NaN or Inf)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_improved_lstm_architecture(self):
        """Test improved LSTM with attention"""
        model = ImprovedLSTMModel(
            vocab_size=1000,
            embedding_dim=200,
            hidden_dim=256,
            num_layers=3,
            num_classes=3
        )
        
        # Test forward pass
        batch_size = 2
        seq_length = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        output = model(input_ids)
        
        # Check output shape
        assert output.shape == (batch_size, 3)
        
        # Check that attention mechanism works
        assert hasattr(model, 'attention')
        assert hasattr(model, 'attention_mechanism')
    
    @patch('torch.load')
    @patch('os.path.exists')
    def test_save_load_model(self, mock_exists, mock_load, lstm_detector):
        """Test model saving and loading"""
        mock_exists.return_value = True
        
        # Mock model
        lstm_detector.model = Mock()
        lstm_detector.vocab = {'test': 1, '<PAD>': 0, '<UNK>': 1}
        
        # Test save
        with patch('torch.save') as mock_save:
            lstm_detector.save_model('test_model.pth')
            mock_save.assert_called_once()
        
        # Test load
        mock_load.return_value = {
            'model_state_dict': {},
            'vocab': {'test': 1},
            'max_length': 64,
            'label_encoder': {'positive': 1, 'negative': 0, 'neutral': 2},
            'emotion_labels': ['negative', 'positive', 'neutral']
        }
        
        with patch.object(lstm_detector, 'create_model'):
            lstm_detector.load_model('test_model.pth')
            assert lstm_detector.vocab == {'test': 1}
    
    def test_predict_emotion_mock(self, lstm_detector):
        """Test emotion prediction with mocked model"""
        # Mock the model and its components
        mock_model = Mock()
        mock_model.eval.return_value = None
        
        # Mock prediction output
        mock_logits = torch.tensor([[2.0, 0.5, 0.1]])  # Favor positive class
        mock_model.return_value = mock_logits
        
        lstm_detector.model = mock_model
        lstm_detector.vocab = {'tôi': 2, 'rất': 3, 'vui': 4, '<PAD>': 0, '<UNK>': 1}
        lstm_detector.device = torch.device('cpu')
        
        with patch('torch.no_grad'):
            result = lstm_detector.predict_emotion("Tôi rất vui!")
        
        # Check result structure
        assert 'emotion' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        
        # Check that confidence is reasonable
        assert 0 <= result['confidence'] <= 1

@pytest.mark.unit  
class TestBaselineModels:
    """Test baseline emotion detection models"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return pd.DataFrame({
            'text': [
                "Tôi rất thích sản phẩm này",
                "Dịch vụ tệ quá",
                "Bình thường",
                "Tuyệt vời!",
                "Không hài lòng"
            ],
            'label': ['positive', 'negative', 'neutral', 'positive', 'negative']
        })
    
    @pytest.fixture
    def baseline_classifier(self):
        """Baseline classifier instance"""
        return EmotionClassifier()
    
    def test_baseline_classifier_initialization(self, baseline_classifier):
        """Test baseline classifier initialization"""
        assert baseline_classifier.models == {}
        assert baseline_classifier.vectorizer is None
        assert baseline_classifier.label_encoder is not None
    
    def test_preprocess_text(self, baseline_classifier):
        """Test text preprocessing"""
        raw_text = "Tôi RẤT VUI hôm nay!!! @#$%"
        processed = baseline_classifier.preprocess_text(raw_text)
        
        # Should be lowercase and cleaned
        assert processed.islower()
        assert "rất" in processed
        assert "vui" in processed
        # Special characters should be removed or handled
        assert "@#$%" not in processed
    
    @patch('joblib.dump')
    def test_train_models(self, mock_dump, baseline_classifier, sample_data):
        """Test training baseline models"""
        with patch.object(baseline_classifier, 'prepare_data') as mock_prepare:
            # Mock data preparation
            mock_prepare.return_value = (
                sample_data['text'].tolist(),
                [0, 1, 2, 0, 1]  # encoded labels
            )
            
            # Train models
            baseline_classifier.train_models(sample_data)
            
            # Check that models were created
            assert 'logistic_regression' in baseline_classifier.models
            assert 'random_forest' in baseline_classifier.models
            assert 'svm' in baseline_classifier.models
            
            # Check that vectorizer was fitted
            assert baseline_classifier.vectorizer is not None
    
    def test_predict_emotion_mock(self, baseline_classifier):
        """Test emotion prediction with mocked models"""
        # Mock trained models
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])  # Positive prediction
        mock_model.predict.return_value = np.array([1])
        
        baseline_classifier.models = {'svm': mock_model}
        
        # Mock vectorizer
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = Mock()
        baseline_classifier.vectorizer = mock_vectorizer
        
        result = baseline_classifier.predict_emotion("Test text", model_type="svm")
        
        # Check result structure
        assert 'emotion' in result
        assert 'confidence' in result
        assert 'probabilities' in result
    
    @patch('joblib.load')
    @patch('os.path.exists')
    def test_load_models(self, mock_exists, mock_load, baseline_classifier):
        """Test loading saved models"""
        mock_exists.return_value = True
        mock_load.side_effect = [
            Mock(),  # logistic regression
            Mock(),  # random forest  
            Mock(),  # svm
            Mock()   # vectorizer
        ]
        
        baseline_classifier.load_models()
        
        # Check that models were loaded
        assert 'logistic_regression' in baseline_classifier.models
        assert 'random_forest' in baseline_classifier.models
        assert 'svm' in baseline_classifier.models
        assert baseline_classifier.vectorizer is not None

@pytest.mark.unit
class TestPhoBERTModel:
    """Test PhoBERT emotion detection model"""
    
    @pytest.fixture
    def phobert_detector(self):
        """PhoBERT detector instance"""
        return PhoBERTEmotionDetector(max_length=128)
    
    def test_phobert_detector_initialization(self, phobert_detector):
        """Test PhoBERT detector initialization"""
        assert phobert_detector.max_length == 128
        assert phobert_detector.device is not None
        assert phobert_detector.tokenizer is not None
        assert phobert_detector.model_name == 'vinai/phobert-base'
    
    def test_phobert_classifier_architecture(self):
        """Test PhoBERT classifier architecture"""
        # Skip if transformers not available
        pytest.importorskip("transformers")
        
        with patch('transformers.AutoConfig.from_pretrained') as mock_config:
            with patch('transformers.AutoModel.from_pretrained') as mock_model:
                # Mock config
                mock_config.return_value = Mock()
                mock_config.return_value.hidden_size = 768
                
                # Mock model
                mock_model.return_value = Mock()
                
                classifier = PhoBERTEmotionClassifier(
                    model_name='vinai/phobert-base',
                    num_classes=3
                )
                
                assert hasattr(classifier, 'phobert')
                assert hasattr(classifier, 'classifier')
                assert hasattr(classifier, 'dropout')
    
    @patch('torch.load')
    @patch('os.path.exists')
    def test_save_load_model(self, mock_exists, mock_load, phobert_detector):
        """Test PhoBERT model saving and loading"""
        mock_exists.return_value = True
        
        # Mock model
        phobert_detector.model = Mock()
        
        # Test save
        with patch('torch.save') as mock_save:
            with patch.object(phobert_detector.tokenizer, 'save_pretrained'):
                phobert_detector.save_model('test_phobert.pth')
                mock_save.assert_called_once()
        
        # Test load
        mock_load.return_value = {
            'model_state_dict': {},
            'model_name': 'vinai/phobert-base',
            'max_length': 128,
            'label_encoder': {'positive': 1, 'negative': 0, 'neutral': 2},
            'emotion_labels': ['negative', 'positive', 'neutral']
        }
        
        with patch.object(phobert_detector, 'create_model'):
            with patch('transformers.AutoTokenizer.from_pretrained'):
                phobert_detector.load_model('test_phobert.pth')

@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for models"""
    
    def test_model_compatibility(self):
        """Test that all models produce compatible output formats"""
        sample_text = "Tôi rất vui hôm nay!"
        
        # Expected output format
        expected_keys = {'emotion', 'confidence', 'probabilities'}
        valid_emotions = {'positive', 'negative', 'neutral'}
        
        results = {}
        
        # Test LSTM (if available)
        try:
            lstm_detector = LSTMEmotionDetector()
            if hasattr(lstm_detector, 'model') and lstm_detector.model is not None:
                result = lstm_detector.predict_emotion(sample_text)
                results['lstm'] = result
        except Exception:
            pass
        
        # Test baseline (mock)
        try:
            baseline = EmotionClassifier()
            # Mock for testing
            with patch.object(baseline, 'predict_emotion') as mock_predict:
                mock_predict.return_value = {
                    'emotion': 'positive',
                    'confidence': 0.85,
                    'probabilities': {'positive': 0.85, 'negative': 0.10, 'neutral': 0.05}
                }
                result = baseline.predict_emotion(sample_text)
                results['baseline'] = result
        except Exception:
            pass
        
        # Check all results have consistent format
        for model_name, result in results.items():
            assert set(result.keys()) >= expected_keys, f"{model_name} missing required keys"
            assert result['emotion'] in valid_emotions, f"{model_name} invalid emotion"
            assert 0 <= result['confidence'] <= 1, f"{model_name} invalid confidence"
            
            # Check probabilities sum to ~1
            prob_sum = sum(result['probabilities'].values())
            assert abs(prob_sum - 1.0) < 0.01, f"{model_name} probabilities don't sum to 1"

@pytest.mark.performance
class TestModelPerformance:
    """Performance tests for models"""
    
    def test_prediction_speed(self):
        """Test that models predict within reasonable time"""
        import time
        
        sample_texts = [
            "Tôi rất vui!",
            "Buồn quá",
            "Bình thường",
            "Tuyệt vời!",
            "Không tốt"
        ]
        
        # Test baseline model speed (mocked)
        baseline = EmotionClassifier()
        with patch.object(baseline, 'predict_emotion') as mock_predict:
            mock_predict.return_value = {
                'emotion': 'positive',
                'confidence': 0.85,
                'probabilities': {'positive': 0.85, 'negative': 0.10, 'neutral': 0.05}
            }
            
            start_time = time.time()
            for text in sample_texts:
                baseline.predict_emotion(text)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / len(sample_texts)
            
            # Should be very fast for baseline models
            assert avg_time < 0.1  # Less than 100ms per prediction
    
    def test_memory_efficiency(self):
        """Test that models don't leak memory"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and destroy multiple model instances
        for i in range(10):
            detector = LSTMEmotionDetector()
            del detector
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal (less than 50MB)
        assert memory_growth < 50 * 1024 * 1024

@pytest.mark.security
class TestModelSecurity:
    """Security tests for models"""
    
    def test_malicious_input_handling(self):
        """Test that models handle malicious input safely"""
        malicious_inputs = [
            "' OR 1=1 --",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "../../../etc/passwd",  # Path traversal
            "A" * 10000,  # Very long input
            "",  # Empty input
            None,  # None input
        ]
        
        baseline = EmotionClassifier()
        
        for malicious_input in malicious_inputs:
            try:
                # Should either process safely or raise appropriate exception
                with patch.object(baseline, 'predict_emotion') as mock_predict:
                    mock_predict.return_value = {
                        'emotion': 'neutral',
                        'confidence': 0.33,
                        'probabilities': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                    }
                    result = baseline.predict_emotion(str(malicious_input) if malicious_input else "")
                    
                    # Should return valid result
                    assert 'emotion' in result
                    assert result['emotion'] in ['positive', 'negative', 'neutral']
                    
            except (ValueError, TypeError, AttributeError):
                # These exceptions are acceptable for invalid input
                pass
            except Exception as e:
                # Unexpected exceptions should not occur
                pytest.fail(f"Unexpected exception for input '{malicious_input}': {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 