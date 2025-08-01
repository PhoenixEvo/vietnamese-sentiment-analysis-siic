#!/usr/bin/env python3
"""
Performance Optimization Script for Vietnamese Emotion Detection System
Optimizes models, implements caching, and provides performance monitoring
"""
import torch
import time
import os
import sys
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import MODELS_DIR, RESULTS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimize models for inference performance"""
    
    def __init__(self):
        self.optimization_results = {}
    
    def optimize_torch_model(self, model_path: str, output_path: str = None) -> Dict[str, Any]:
        """Optimize PyTorch model for inference"""
        logger.info(f"Optimizing PyTorch model: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return {"success": False, "error": "Model file not found"}
        
        try:
            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device)
            
            # Create optimized model path
            if output_path is None:
                base_name = os.path.splitext(model_path)[0]
                output_path = f"{base_name}_optimized.pth"
            
            # Optimization techniques
            optimizations = []
            original_size = os.path.getsize(model_path)
            
            # 1. Model compression (if model state dict is available)
            if 'model_state_dict' in checkpoint:
                logger.info("  Applying model compression...")
                
                # Half precision optimization (if CUDA available)
                if torch.cuda.is_available():
                    try:
                        model_state = checkpoint['model_state_dict']
                        for key in model_state:
                            if model_state[key].dtype == torch.float32:
                                model_state[key] = model_state[key].half()
                        optimizations.append("half_precision")
                        logger.info("  Applied half precision optimization")
                    except Exception as e:
                        logger.warning(f"  Half precision failed: {e}")
                
                # Remove unnecessary training parameters
                training_keys = [k for k in checkpoint.keys() if 'optim' in k.lower() or 'scheduler' in k.lower()]
                for key in training_keys:
                    if key in checkpoint:
                        del checkpoint[key]
                        optimizations.append(f"removed_{key}")
                
                logger.info(f"  Removed {len(training_keys)} training parameters")
            
            # 2. Save optimized model
            torch.save(checkpoint, output_path)
            optimized_size = os.path.getsize(output_path)
            
            # Calculate compression ratio
            compression_ratio = (original_size - optimized_size) / original_size * 100
            
            result = {
                "success": True,
                "original_size": original_size,
                "optimized_size": optimized_size,
                "compression_ratio": compression_ratio,
                "optimizations": optimizations,
                "output_path": output_path
            }
            
            logger.info(f"  Model optimized: {compression_ratio:.1f}% size reduction")
            return result
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return {"success": False, "error": str(e)}
    
    def benchmark_model_inference(self, model_path: str, num_samples: int = 100) -> Dict[str, Any]:
        """Benchmark model inference speed"""
        logger.info(f"Benchmarking model inference: {model_path}")
        
        try:
            # Prepare sample data
            sample_texts = [
                "Tôi rất vui hôm nay!",
                "Buồn quá, không muốn làm gì",
                "Sản phẩm này tuyệt vời",
                "Dịch vụ khách hàng tệ",
                "Bình thường thôi"
            ] * (num_samples // 5 + 1)
            sample_texts = sample_texts[:num_samples]
            
            # Import appropriate model class
            if 'lstm' in model_path.lower():
                from src.models.lstm_model import LSTMEmotionDetector
                model = LSTMEmotionDetector()
                model.load_model(model_path)
            
                model.load_model(model_path)
            else:
                logger.warning(f"Unknown model type: {model_path}")
                return {"success": False, "error": "Unknown model type"}
            
            # Warmup
            logger.info("  Warming up model...")
            for i in range(5):
                model.predict_emotion(sample_texts[i])
            
            # Benchmark
            logger.info(f"  Running {num_samples} predictions...")
            start_time = time.time()
            
            prediction_times = []
            for text in sample_texts:
                pred_start = time.time()
                result = model.predict_emotion(text)
                pred_end = time.time()
                prediction_times.append(pred_end - pred_start)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate statistics
            avg_time = np.mean(prediction_times)
            min_time = np.min(prediction_times)
            max_time = np.max(prediction_times)
            p95_time = np.percentile(prediction_times, 95)
            throughput = num_samples / total_time
            
            result = {
                "success": True,
                "num_samples": num_samples,
                "total_time": total_time,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "p95_time": p95_time,
                "throughput": throughput,
                "predictions_per_second": throughput
            }
            
            logger.info(f"  Benchmark complete: {throughput:.1f} predictions/sec")
            return result
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {"success": False, "error": str(e)}

class CacheManager:
    """Implement intelligent caching for predictions"""
    
    def __init__(self, cache_dir: str = None, max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache_file = self.cache_dir / "prediction_cache.pkl"
        self.stats_file = self.cache_dir / "cache_stats.json"
        
        # Load existing cache
        self.cache = self.load_cache()
        self.stats = self.load_stats()
    
    def get_text_hash(self, text: str, model_type: str) -> str:
        """Generate hash for text and model combination"""
        combined = f"{text.strip().lower()}:{model_type}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, text: str, model_type: str) -> Dict[str, Any]:
        """Get cached prediction"""
        cache_key = self.get_text_hash(text, model_type)
        
        if cache_key in self.cache:
            self.stats['hits'] += 1
            logger.debug(f"Cache hit for key: {cache_key[:8]}...")
            return self.cache[cache_key]['result']
        
        self.stats['misses'] += 1
        return None
    
    def set(self, text: str, model_type: str, result: Dict[str, Any]) -> None:
        """Cache prediction result"""
        cache_key = self.get_text_hash(text, model_type)
        
        cache_entry = {
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'model_type': model_type
        }
        
        self.cache[cache_key] = cache_entry
        self.stats['entries'] = len(self.cache)
        
        # Clean cache if it's too large
        self.cleanup_cache()
        
        logger.debug(f"Cached result for key: {cache_key[:8]}...")
    
    def cleanup_cache(self) -> None:
        """Remove old entries if cache is too large"""
        # Estimate cache size
        estimated_size = len(str(self.cache).encode())
        
        if estimated_size > self.max_size_bytes:
            logger.info(" Cleaning up cache...")
            
            # Sort by timestamp (oldest first)
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            # Remove oldest 25% of entries
            num_to_remove = len(sorted_items) // 4
            for i in range(num_to_remove):
                key = sorted_items[i][0]
                del self.cache[key]
            
            self.stats['cleanups'] += 1
            logger.info(f" Removed {num_to_remove} old cache entries")
    
    def save_cache(self) -> None:
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            self.save_stats()
            logger.debug("Cache saved to disk")
        except Exception as e:
            logger.error(f" Failed to save cache: {e}")
    
    def load_cache(self) -> Dict[str, Any]:
        """Load cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f" Loaded cache with {len(cache)} entries")
                return cache
        except Exception as e:
            logger.warning(f" Failed to load cache: {e}")
        
        return {}
    
    def load_stats(self) -> Dict[str, int]:
        """Load cache statistics"""
        try:
            if self.stats_file.exists():
                import json
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f" Failed to load cache stats: {e}")
        
        return {
            'hits': 0,
            'misses': 0,
            'entries': 0,
            'cleanups': 0
        }
    
    def save_stats(self) -> None:
        """Save cache statistics"""
        try:
            import json
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f" Failed to save cache stats: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'total_entries': len(self.cache),
            'total_requests': total_requests,
            'cache_hits': self.stats['hits'],
            'cache_misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'cleanups': self.stats['cleanups']
        }

class PerformanceMonitor:
    """Monitor and analyze system performance"""
    
    def __init__(self):
        self.metrics = {
            'predictions': [],
            'response_times': [],
            'memory_usage': [],
            'cpu_usage': []
        }
    
    def record_prediction(self, processing_time: float, memory_mb: float = None) -> None:
        """Record a prediction event"""
        self.metrics['predictions'].append({
            'timestamp': datetime.now(),
            'processing_time': processing_time,
            'memory_mb': memory_mb
        })
        
        self.metrics['response_times'].append(processing_time)
        
        if memory_mb:
            self.metrics['memory_usage'].append(memory_mb)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        if not self.metrics['response_times']:
            return {"error": "No performance data available"}
        
        response_times = self.metrics['response_times']
        
        report = {
            'total_predictions': len(response_times),
            'avg_response_time': np.mean(response_times),
            'min_response_time': np.min(response_times),
            'max_response_time': np.max(response_times),
            'p50_response_time': np.percentile(response_times, 50),
            'p95_response_time': np.percentile(response_times, 95),
            'p99_response_time': np.percentile(response_times, 99),
            'predictions_per_second': len(response_times) / (len(response_times) * np.mean(response_times)),
        }
        
        if self.metrics['memory_usage']:
            memory_usage = self.metrics['memory_usage']
            report.update({
                'avg_memory_usage_mb': np.mean(memory_usage),
                'max_memory_usage_mb': np.max(memory_usage),
                'min_memory_usage_mb': np.min(memory_usage)
            })
        
        return report
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to file"""
        try:
            import json
            
            # Convert datetime objects to strings
            exportable_metrics = {}
            for key, values in self.metrics.items():
                if key == 'predictions':
                    exportable_metrics[key] = [
                        {
                            'timestamp': item['timestamp'].isoformat(),
                            'processing_time': item['processing_time'],
                            'memory_mb': item.get('memory_mb')
                        }
                        for item in values
                    ]
                else:
                    exportable_metrics[key] = values
            
            with open(filepath, 'w') as f:
                json.dump(exportable_metrics, f, indent=2)
            
            logger.info(f" Metrics exported to: {filepath}")
            
        except Exception as e:
            logger.error(f" Failed to export metrics: {e}")

class SystemOptimizer:
    """Comprehensive system optimization"""
    
    def __init__(self):
        self.model_optimizer = ModelOptimizer()
        self.cache_manager = CacheManager()
        self.performance_monitor = PerformanceMonitor()
    
    def optimize_all_models(self) -> Dict[str, Any]:
        """Optimize all available models"""
        logger.info("Starting comprehensive model optimization...")
        
        models_dir = Path(MODELS_DIR)
        optimization_results = {}
        
        # Find all PyTorch model files
        model_files = list(models_dir.glob("*.pth"))
        
        for model_file in model_files:
            logger.info(f"🔧 Processing model: {model_file.name}")
            
            # Optimize model
            opt_result = self.model_optimizer.optimize_torch_model(str(model_file))
            
            # Benchmark model
            if opt_result.get('success'):
                benchmark_result = self.model_optimizer.benchmark_model_inference(
                    opt_result['output_path'], num_samples=50
                )
                opt_result['benchmark'] = benchmark_result
            
            optimization_results[model_file.name] = opt_result
        
        return optimization_results
    
    def setup_production_environment(self) -> Dict[str, Any]:
        """Setup optimized production environment"""
        logger.info("🏭 Setting up production environment...")
        
        results = {
            'torch_settings': {},
            'environment_variables': {},
            'optimizations': []
        }
        
        try:
            # PyTorch optimizations
            if torch.cuda.is_available():
                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                results['torch_settings']['cudnn_benchmark'] = True
                results['optimizations'].append('cuda_optimizations')
                logger.info("   Enabled CUDA optimizations")
            
            # CPU optimizations
            torch.set_num_threads(os.cpu_count())
            results['torch_settings']['num_threads'] = os.cpu_count()
            results['optimizations'].append('cpu_threading')
            logger.info(f"   Set PyTorch threads: {os.cpu_count()}")
            
            # Memory optimizations
            if hasattr(torch.backends, 'mkldnn'):
                torch.backends.mkldnn.enabled = True
                results['optimizations'].append('mkldnn')
                logger.info("   Enabled MKL-DNN optimizations")
            
            # Environment variables for production
            production_env = {
                'OMP_NUM_THREADS': str(os.cpu_count()),
                'TOKENIZERS_PARALLELISM': 'false',  # Avoid warnings in production
                'PYTHONUNBUFFERED': '1',  # Ensure proper logging
            }
            
            for key, value in production_env.items():
                os.environ[key] = value
                results['environment_variables'][key] = value
            
            logger.info("   Set production environment variables")
            
            results['success'] = True
            return results
            
        except Exception as e:
            logger.error(f" Production setup failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            return results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        logger.info(" Generating performance report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.get_system_info(),
            'cache_stats': self.cache_manager.get_cache_stats(),
            'performance_metrics': self.performance_monitor.get_performance_report(),
            'optimization_recommendations': self.get_optimization_recommendations()
        }
        
        return report
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import platform
        import psutil
        
        try:
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': os.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'pytorch_version': torch.__version__
            }
        except ImportError:
            logger.warning(" psutil not available for detailed system info")
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': os.cpu_count(),
                'cuda_available': torch.cuda.is_available(),
                'pytorch_version': torch.__version__
            }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current setup"""
        recommendations = []
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            recommendations.append("Consider using GPU for better performance")
        
        # Check CPU count
        if os.cpu_count() < 4:
            recommendations.append("Consider upgrading to a multi-core CPU")
        
        # Check cache hit rate
        cache_stats = self.cache_manager.get_cache_stats()
        if cache_stats['hit_rate'] < 0.3:
            recommendations.append("Cache hit rate is low - consider increasing cache size")
        
        # Check memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                recommendations.append("High memory usage - consider adding more RAM")
        except ImportError:
            pass
        
        if not recommendations:
            recommendations.append("System is well optimized!")
        
        return recommendations

def main():
    """Main optimization function"""
    logger.info("Starting Vietnamese Emotion Detection System Optimization")
    
    # Initialize system optimizer
    optimizer = SystemOptimizer()
    
    # Setup production environment
    logger.info("\ Step 1: Production Environment Setup")
    env_result = optimizer.setup_production_environment()
    
    if env_result['success']:
        logger.info(" Production environment setup complete")
    else:
        logger.error(f" Production setup failed: {env_result.get('error')}")
    
    # Optimize models
    logger.info("\ Step 2: Model Optimization")
    model_results = optimizer.optimize_all_models()
    
    total_models = len(model_results)
    successful_optimizations = sum(1 for r in model_results.values() if r.get('success'))
    
    logger.info(f"Models optimized: {successful_optimizations}/{total_models}")
    
    # Generate performance report
    logger.info("\ Step 3: Performance Analysis")
    performance_report = optimizer.generate_performance_report()
    
    # Save results
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    
    # Save optimization results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"optimization_results_{timestamp}.json"
    
    final_results = {
        'environment_setup': env_result,
        'model_optimizations': model_results,
        'performance_report': performance_report,
        'timestamp': timestamp
    }
    
    try:
        import json
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f" Results saved to: {results_file}")
    except Exception as e:
        logger.error(f" Failed to save results: {e}")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info(" OPTIMIZATION SUMMARY")
    logger.info("="*50)
    
    logger.info(f"Models optimized: {successful_optimizations}/{total_models}")
    
    if successful_optimizations > 0:
        total_compression = sum(
            r.get('compression_ratio', 0) 
            for r in model_results.values() 
            if r.get('success')
        ) / successful_optimizations
        logger.info(f"📦 Average compression: {total_compression:.1f}%")
    
    cache_stats = optimizer.cache_manager.get_cache_stats()
    logger.info(f"Cache hit rate: {cache_stats['hit_rate']:.1%}")
    
    recommendations = performance_report['optimization_recommendations']
    logger.info(f"Recommendations: {len(recommendations)}")
    for rec in recommendations:
        logger.info(f"  • {rec}")
    
    logger.info("\n Optimization complete!")
    
    return final_results

if __name__ == "__main__":
    main() 