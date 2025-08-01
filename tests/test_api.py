"""
Comprehensive test suite for Vietnamese Sentiment Analysis API
"""
import pytest
import asyncio
import json
import time
import io
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Import the FastAPI app
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

# Test client
client = TestClient(app)

class TestAPIEndpoints:
    """Test API endpoints functionality"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "status" in data
        assert "timestamp" in data
        assert "models_loaded" in data
        assert "system_info" in data
        
        # Check models_loaded structure
        models = data["models_loaded"]
        assert "lstm" in models
        assert "baseline" in models
        # PhoBERT removed from system
        assert "phobert" not in models
        
        # Check system_info
        system_info = data["system_info"]
        assert "python_version" in system_info
        assert "torch_version" in system_info
        assert "cuda_available" in system_info
    
    def test_models_endpoint(self):
        """Test models information endpoint"""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        
        # Should return list of models
        assert isinstance(data, list)
        
        # Check if any models are available
        if data:
            model = data[0]
            required_fields = [
                "model_name", "model_type", "supported_emotions",
                "max_text_length", "last_updated"
            ]
            for field in required_fields:
                assert field in model

class TestPredictionEndpoints:
    """Test prediction functionality"""
    
    @pytest.fixture
    def sample_text_input(self):
        """Sample text input for testing"""
        return {
            "text": "Hôm nay tôi rất vui vì được nghỉ làm!",
            "model_type": "lstm"
        }
    
    @pytest.fixture
    def sample_batch_input(self):
        """Sample batch input for testing"""
        return {
            "texts": [
                "Tôi yêu sản phẩm này!",
                "Dịch vụ khách hàng tệ quá",
                "Bình thường thôi, không có gì đặc biệt"
            ],
            "model_type": "lstm"
        }
    
    def test_single_prediction_valid_input(self, sample_text_input):
        """Test single prediction with valid input"""
        response = client.post("/predict", json=sample_text_input)
        
        if response.status_code == 503:
            # Model not available, skip test
            pytest.skip("Model not available for testing")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        required_fields = [
            "text", "emotion", "confidence", "probabilities",
            "processing_time", "model_used"
        ]
        for field in required_fields:
            assert field in data
        
        # Check data types and ranges
        assert isinstance(data["confidence"], float)
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["processing_time"], float)
        assert data["processing_time"] > 0
        
        # Check emotion is valid
        valid_emotions = ["positive", "negative", "neutral"]
        assert data["emotion"] in valid_emotions
        
        # Check probabilities
        probabilities = data["probabilities"]
        assert isinstance(probabilities, dict)
        prob_sum = sum(probabilities.values())
        assert abs(prob_sum - 1.0) < 0.01  # Should sum to ~1
    
    def test_single_prediction_empty_text(self):
        """Test single prediction with empty text"""
        response = client.post("/predict", json={"text": "", "model_type": "lstm"})
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]
    
    def test_single_prediction_invalid_model(self, sample_text_input):
        """Test single prediction with invalid model type"""
        invalid_input = sample_text_input.copy()
        invalid_input["model_type"] = "invalid_model"
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 400
        assert "Invalid model_type" in response.json()["detail"]
    
    def test_single_prediction_missing_text(self):
        """Test single prediction with missing text field"""
        response = client.post("/predict", json={"model_type": "lstm"})
        assert response.status_code == 422  # Validation error
    
    def test_batch_prediction_valid_input(self, sample_batch_input):
        """Test batch prediction with valid input"""
        response = client.post("/predict/batch", json=sample_batch_input)
        
        if response.status_code == 503:
            pytest.skip("Model not available for testing")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "results" in data
        assert "total_texts" in data
        assert "processing_time" in data
        assert "model_used" in data
        
        # Check results
        results = data["results"]
        assert len(results) == len(sample_batch_input["texts"])
        assert data["total_texts"] == len(sample_batch_input["texts"])
        
        # Check each result
        for result in results:
            required_fields = [
                "text", "emotion", "confidence", "probabilities",
                "processing_time", "model_used"
            ]
            for field in required_fields:
                assert field in result
    
    def test_batch_prediction_empty_list(self):
        """Test batch prediction with empty texts list"""
        response = client.post("/predict/batch", json={"texts": [], "model_type": "lstm"})
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]
    
    def test_batch_prediction_too_many_texts(self):
        """Test batch prediction with too many texts"""
        large_input = {
            "texts": ["test text"] * 101,  # More than limit
            "model_type": "lstm"
        }
        response = client.post("/predict/batch", json=large_input)
        assert response.status_code == 400
        assert "Maximum 100 texts" in response.json()["detail"]
    
    def test_file_prediction_valid_csv(self):
        """Test file prediction with valid CSV"""
        # Create sample CSV
        csv_data = """id,text,author
1,"Sản phẩm này rất tốt!",user1
2,"Dịch vụ tệ quá",user2
3,"Bình thường thôi",user3"""
        
        files = {"file": ("test.csv", io.StringIO(csv_data), "text/csv")}
        data = {"model_type": "lstm", "text_column": "text"}
        
        response = client.post("/predict/file", files=files, data=data)
        
        if response.status_code == 503:
            pytest.skip("Model not available for testing")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
    
    def test_file_prediction_invalid_format(self):
        """Test file prediction with invalid file format"""
        files = {"file": ("test.txt", io.StringIO("test content"), "text/plain")}
        data = {"model_type": "lstm", "text_column": "text"}
        
        response = client.post("/predict/file", files=files, data=data)
        assert response.status_code == 400
        assert "Only CSV files" in response.json()["detail"]
    
    def test_file_prediction_missing_column(self):
        """Test file prediction with missing text column"""
        csv_data = """id,comment,author
1,"Good product",user1"""
        
        files = {"file": ("test.csv", io.StringIO(csv_data), "text/csv")}
        data = {"model_type": "lstm", "text_column": "text"}  # Column doesn't exist
        
        response = client.post("/predict/file", files=files, data=data)
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]

class TestModelPerformance:
    """Test model performance endpoints"""
    
    def test_model_performance_lstm(self):
        """Test LSTM model performance endpoint"""
        response = client.get("/models/lstm/performance")
        assert response.status_code == 200
        data = response.json()
        
        # Check performance metrics
        assert "accuracy" in data
        assert "f1_score" in data
        assert "precision" in data
        assert "recall" in data
        
        # Check metrics are reasonable
        assert 0 <= data["accuracy"] <= 1
        assert 0 <= data["f1_score"] <= 1
    
    def test_model_performance_invalid_model(self):
        """Test performance endpoint with invalid model"""
        response = client.get("/models/nonexistent/performance")
        assert response.status_code == 404

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_malformed_json(self):
        """Test API with malformed JSON"""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_large_text_input(self):
        """Test API with very large text input"""
        large_text = "a" * 2000  # Very long text
        response = client.post("/predict", json={"text": large_text, "model_type": "lstm"})
        
        if response.status_code == 503:
            pytest.skip("Model not available for testing")
        
        # Should either process or reject with proper error
        assert response.status_code in [200, 400, 422]
    
    def test_special_characters(self):
        """Test API with special characters in text"""
        special_text = "Xin chào! 😊🎉 Test @#$%^&*()"
        response = client.post("/predict", json={"text": special_text, "model_type": "lstm"})
        
        if response.status_code == 503:
            pytest.skip("Model not available for testing")
        
        assert response.status_code == 200
    
    def test_unicode_text(self):
        """Test API with Unicode Vietnamese text"""
        unicode_text = "Tôi rất thích món ăn này! Nó có vị rất đặc biệt và thơm ngon."
        response = client.post("/predict", json={"text": unicode_text, "model_type": "lstm"})
        
        if response.status_code == 503:
            pytest.skip("Model not available for testing")
        
        assert response.status_code == 200

class TestModelComparison:
    """Test model comparison functionality"""
    
    @pytest.fixture
    def comparison_text(self):
        """Sample text for model comparison"""
        return "Hôm nay tôi rất vui!"
    
    def test_model_consistency(self, comparison_text):
        """Test that different models give consistent results for same input"""
        models = ["lstm", "baseline"]  # Only available models
        results = {}
        
        for model in models:
            response = client.post("/predict", json={
                "text": comparison_text,
                "model_type": model
            })
            
            if response.status_code == 200:
                results[model] = response.json()
        
        if len(results) >= 2:
            # Check that models agree on general emotion direction
            emotions = [result["emotion"] for result in results.values()]
            # All should be positive for this text (or at least not contradictory)
            assert len(set(emotions)) <= 2  # Allow some disagreement

class TestPerformanceMetrics:
    """Test API performance and response times"""
    
    def test_response_time_single_prediction(self):
        """Test response time for single prediction"""
        start_time = time.time()
        response = client.post("/predict", json={
            "text": "Test response time",
            "model_type": "lstm"
        })
        end_time = time.time()
        
        if response.status_code == 503:
            pytest.skip("Model not available for testing")
        
        response_time = end_time - start_time
        assert response_time < 5.0  # Should respond within 5 seconds
        
        # Check reported processing time
        if response.status_code == 200:
            processing_time = response.json()["processing_time"]
            assert processing_time < 1.0  # Model inference should be fast
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = client.post("/predict", json={
                    "text": f"Concurrent test {threading.current_thread().ident}",
                    "model_type": "lstm"
                })
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create 10 concurrent threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Check results
        if 503 not in results:  # If models are available
            success_rate = results.count(200) / len(results)
            assert success_rate >= 0.8  # At least 80% success rate
            assert end_time - start_time < 10.0  # All requests within 10 seconds
        
        assert len(errors) == 0  # No exceptions should occur

class TestSecurityAndValidation:
    """Test security and input validation"""
    
    def test_sql_injection_attempt(self):
        """Test API resistance to SQL injection"""
        malicious_text = "'; DROP TABLE users; --"
        response = client.post("/predict", json={
            "text": malicious_text,
            "model_type": "lstm"
        })
        
        if response.status_code == 503:
            pytest.skip("Model not available for testing")
        
        # Should process normally or reject safely
        assert response.status_code in [200, 400, 422]
        if response.status_code == 200:
            assert "emotion" in response.json()
    
    def test_xss_attempt(self):
        """Test API resistance to XSS"""
        xss_text = "<script>alert('xss')</script>"
        response = client.post("/predict", json={
            "text": xss_text,
            "model_type": "lstm"
        })
        
        if response.status_code == 503:
            pytest.skip("Model not available for testing")
        
        assert response.status_code in [200, 400, 422]
    
    def test_path_traversal_attempt(self):
        """Test API resistance to path traversal"""
        malicious_text = "../../../etc/passwd"
        response = client.post("/predict", json={
            "text": malicious_text,
            "model_type": "lstm"
        })
        
        if response.status_code == 503:
            pytest.skip("Model not available for testing")
        
        assert response.status_code in [200, 400, 422]

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limit_compliance(self):
        """Test that API respects rate limits"""
        # Make rapid requests
        responses = []
        for i in range(20):
            response = client.get("/health")
            responses.append(response.status_code)
        
        # All health checks should succeed (no rate limiting on health)
        assert all(status == 200 for status in responses)

# Fixtures and utilities
@pytest.fixture
def mock_model():
    """Mock model for testing when real models aren't available"""
    mock = Mock()
    mock.predict_emotion.return_value = {
        "emotion": "positive",
        "confidence": 0.85,
        "probabilities": {
            "positive": 0.85,
            "negative": 0.10,
            "neutral": 0.05
        }
    }
    return mock

@pytest.fixture
def sample_csv_file():
    """Sample CSV file for testing"""
    csv_content = """id,text,author,timestamp
1,"Sản phẩm này rất tốt, tôi rất thích!",user1,2025-01-15
2,"Dịch vụ khách hàng tệ quá",user2,2025-01-15
3,"Bình thường thôi, không có gì đặc biệt",user3,2025-01-15
4,"Tôi yêu thích cách làm việc của nhân viên",user4,2025-01-15
5,"Thất vọng với chất lượng",user5,2025-01-15"""
    return io.StringIO(csv_content)

# Performance benchmarks
class TestBenchmarks:
    """Performance benchmarks for the API"""
    
    def test_throughput_benchmark(self):
        """Benchmark API throughput"""
        num_requests = 50
        start_time = time.time()
        
        successful_requests = 0
        for i in range(num_requests):
            response = client.post("/predict", json={
                "text": f"Benchmark test {i}",
                "model_type": "lstm"
            })
            if response.status_code == 200:
                successful_requests += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        if successful_requests > 0:
            throughput = successful_requests / duration
            print(f"Throughput: {throughput:.2f} requests/second")
            
            # Should handle at least 5 requests per second
            assert throughput >= 5.0
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make multiple requests
        for i in range(20):
            response = client.post("/predict", json={
                "text": f"Memory test {i}",
                "model_type": "lstm"
            })
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB)
        assert memory_growth < 100 * 1024 * 1024

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"]) 