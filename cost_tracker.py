"""Cost tracking and profiling utilities for patent classification."""

import time
import psutil
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import logging
from contextlib import contextmanager
import threading
import torch

logger = logging.getLogger(__name__)

@dataclass
class CostMetrics:
    """Data class for storing cost and performance metrics."""
    # Timing metrics
    total_runtime: float = 0.0
    model_loading_time: float = 0.0
    inference_time: float = 0.0
    preprocessing_time: float = 0.0
    
    # Token metrics
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    
    # Resource usage
    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    avg_gpu_memory_mb: float = 0.0
    
    # Cost estimates (based on common pricing)
    estimated_compute_cost_usd: float = 0.0
    estimated_token_cost_usd: float = 0.0
    total_estimated_cost_usd: float = 0.0
    
    # Performance metrics
    samples_processed: int = 0
    samples_per_second: float = 0.0
    cost_per_sample_usd: float = 0.0
    
    # Configuration used
    model_name: str = ""
    batch_size: int = 0
    max_length: int = 0
    quantization: str = ""
    
    # Timestamp
    timestamp: str = ""

class CostTracker:
    """Tracks costs and resource usage during model inference."""
    
    def __init__(self, model_name: str = "", track_gpu: bool = True):
        self.model_name = model_name
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.metrics = CostMetrics()
        
        # Timing tracking
        self._start_time = None
        self._model_loading_start = None
        self._inference_start = None
        
        # Resource monitoring
        self._monitoring = False
        self._monitor_thread = None
        self._cpu_samples = []
        self._memory_samples = []
        self._gpu_memory_samples = []
        
        # Token tracking
        self._sample_input_tokens = []
        self._sample_output_tokens = []
        
        # Cost configuration (approximate pricing)
        self.cost_config = {
            # GPU compute costs (per hour, approximate)
            "gpu_costs": {
                "T4": 0.35,      # Google Colab Pro
                "V100": 2.48,    # AWS p3.2xlarge
                "A100": 3.06,    # AWS p4d.xlarge
                "RTX_3080": 0.50, # Estimated local cost
                "RTX_4090": 0.80, # Estimated local cost
                "default": 1.0    # Default fallback
            },
            # Token costs (per 1M tokens, approximate for hosted inference)
            "token_costs": {
                "llama-8b": {"input": 0.15, "output": 0.60},  # Rough estimate
                "default": {"input": 0.50, "output": 2.00}    # Conservative estimate
            }
        }
    
    @contextmanager
    def track_model_loading(self):
        """Context manager for tracking model loading time."""
        self._model_loading_start = time.time()
        yield
        if self._model_loading_start:
            self.metrics.model_loading_time = time.time() - self._model_loading_start
    
    @contextmanager
    def track_inference(self):
        """Context manager for tracking inference time."""
        self._inference_start = time.time()
        yield
        if self._inference_start:
            self.metrics.inference_time += time.time() - self._inference_start
    
    def start_run(self):
        """Start tracking a full run."""
        self._start_time = time.time()
        self.metrics.timestamp = datetime.now().isoformat()
        self.start_resource_monitoring()
        
    def end_run(self):
        """End tracking and calculate final metrics."""
        if self._start_time:
            self.metrics.total_runtime = time.time() - self._start_time
            
        self.stop_resource_monitoring()
        self._calculate_final_metrics()
    
    def start_resource_monitoring(self):
        """Start background resource monitoring."""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
    
    def stop_resource_monitoring(self):
        """Stop background resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self):
        """Background monitoring of system resources."""
        while self._monitoring:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / 1024 / 1024
                
                self._cpu_samples.append(cpu_percent)
                self._memory_samples.append(memory_mb)
                
                # GPU Memory (if available)
                if self.track_gpu:
                    try:
                        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                        self._gpu_memory_samples.append(gpu_memory_mb)
                    except:
                        pass
                        
                time.sleep(1.0)  # Sample every second
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
    
    def add_token_count(self, input_tokens: int, output_tokens: int):
        """Add token counts for a single inference."""
        self._sample_input_tokens.append(input_tokens)
        self._sample_output_tokens.append(output_tokens)
        self.metrics.total_input_tokens += input_tokens
        self.metrics.total_output_tokens += output_tokens
    
    def add_sample_processed(self):
        """Increment the count of processed samples."""
        self.metrics.samples_processed += 1
    
    def _calculate_final_metrics(self):
        """Calculate final aggregated metrics."""
        # Resource usage averages and peaks
        if self._cpu_samples:
            self.metrics.avg_cpu_percent = sum(self._cpu_samples) / len(self._cpu_samples)
            self.metrics.peak_cpu_percent = max(self._cpu_samples)
            
        if self._memory_samples:
            self.metrics.avg_memory_mb = sum(self._memory_samples) / len(self._memory_samples)
            self.metrics.peak_memory_mb = max(self._memory_samples)
            
        if self._gpu_memory_samples:
            self.metrics.avg_gpu_memory_mb = sum(self._gpu_memory_samples) / len(self._gpu_memory_samples)
            self.metrics.peak_gpu_memory_mb = max(self._gpu_memory_samples)
        
        # Token averages
        if self.metrics.samples_processed > 0:
            self.metrics.avg_input_tokens = self.metrics.total_input_tokens / self.metrics.samples_processed
            self.metrics.avg_output_tokens = self.metrics.total_output_tokens / self.metrics.samples_processed
            
            # Performance metrics
            if self.metrics.total_runtime > 0:
                self.metrics.samples_per_second = self.metrics.samples_processed / self.metrics.total_runtime
        
        # Cost estimation
        self._estimate_costs()
    
    def _estimate_costs(self):
        """Estimate total costs based on usage."""
        # GPU compute cost (if using GPU)
        if self.track_gpu and self.metrics.total_runtime > 0:
            gpu_type = self._detect_gpu_type()
            gpu_cost_per_hour = self.cost_config["gpu_costs"].get(gpu_type, self.cost_config["gpu_costs"]["default"])
            runtime_hours = self.metrics.total_runtime / 3600
            self.metrics.estimated_compute_cost_usd = gpu_cost_per_hour * runtime_hours
        
        # Token cost (for hosted inference)
        model_key = "llama-8b" if "llama" in self.model_name.lower() else "default"
        token_costs = self.cost_config["token_costs"].get(model_key, self.cost_config["token_costs"]["default"])
        
        input_cost = (self.metrics.total_input_tokens / 1_000_000) * token_costs["input"]
        output_cost = (self.metrics.total_output_tokens / 1_000_000) * token_costs["output"]
        self.metrics.estimated_token_cost_usd = input_cost + output_cost
        
        # Total cost
        self.metrics.total_estimated_cost_usd = self.metrics.estimated_compute_cost_usd + self.metrics.estimated_token_cost_usd
        
        # Cost per sample
        if self.metrics.samples_processed > 0:
            self.metrics.cost_per_sample_usd = self.metrics.total_estimated_cost_usd / self.metrics.samples_processed
    
    def _detect_gpu_type(self) -> str:
        """Detect GPU type for cost estimation."""
        if not torch.cuda.is_available():
            return "default"
            
        try:
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "t4" in gpu_name:
                return "T4"
            elif "v100" in gpu_name:
                return "V100"
            elif "a100" in gpu_name:
                return "A100"
            elif "3080" in gpu_name:
                return "RTX_3080"
            elif "4090" in gpu_name:
                return "RTX_4090"
        except:
            pass
            
        return "default"
    
    def update_config_info(self, batch_size: int, max_length: int, quantization: str):
        """Update configuration information."""
        self.metrics.model_name = self.model_name
        self.metrics.batch_size = batch_size
        self.metrics.max_length = max_length
        self.metrics.quantization = quantization
    
    def get_metrics(self) -> CostMetrics:
        """Get current metrics."""
        return self.metrics
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self.metrics), f, indent=2)
            
    def print_cost_summary(self):
        """Print a formatted cost summary."""
        print("\n" + "="*60)
        print("💰 COST PROFILING SUMMARY")
        print("="*60)
        
        print(f"\n⏱️  TIMING:")
        print(f"  Total Runtime: {self.metrics.total_runtime:.2f}s")
        print(f"  Model Loading: {self.metrics.model_loading_time:.2f}s")
        print(f"  Inference Time: {self.metrics.inference_time:.2f}s")
        
        print(f"\n🔢 TOKENS:")
        print(f"  Total Input Tokens: {self.metrics.total_input_tokens:,}")
        print(f"  Total Output Tokens: {self.metrics.total_output_tokens:,}")
        print(f"  Avg Input/Sample: {self.metrics.avg_input_tokens:.1f}")
        print(f"  Avg Output/Sample: {self.metrics.avg_output_tokens:.1f}")
        
        print(f"\n💻 RESOURCES:")
        print(f"  Peak CPU: {self.metrics.peak_cpu_percent:.1f}%")
        print(f"  Peak Memory: {self.metrics.peak_memory_mb:.1f} MB")
        if self.track_gpu:
            print(f"  Peak GPU Memory: {self.metrics.peak_gpu_memory_mb:.1f} MB")
        
        print(f"\n📊 PERFORMANCE:")
        print(f"  Samples Processed: {self.metrics.samples_processed}")
        print(f"  Samples/Second: {self.metrics.samples_per_second:.2f}")
        
        print(f"\n💵 ESTIMATED COSTS:")
        print(f"  Compute Cost: ${self.metrics.estimated_compute_cost_usd:.4f}")
        print(f"  Token Cost: ${self.metrics.estimated_token_cost_usd:.4f}")
        print(f"  Total Cost: ${self.metrics.total_estimated_cost_usd:.4f}")
        print(f"  Cost/Sample: ${self.metrics.cost_per_sample_usd:.6f}")
        
        print(f"\n⚙️  CONFIG:")
        print(f"  Model: {self.metrics.model_name}")
        print(f"  Batch Size: {self.metrics.batch_size}")
        print(f"  Max Length: {self.metrics.max_length}")
        print(f"  Quantization: {self.metrics.quantization}")

class CostComparator:
    """Compare costs across different runs for optimization."""
    
    @staticmethod
    def load_metrics_from_file(filepath: str) -> CostMetrics:
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            return CostMetrics(**data)
    
    @staticmethod
    def compare_runs(metrics_list: List[CostMetrics], labels: List[str] = None) -> Dict:
        """Compare multiple runs and identify best configurations."""
        if not metrics_list:
            return {}
            
        if labels is None:
            labels = [f"Run {i+1}" for i in range(len(metrics_list))]
        
        comparison = {
            "summary": {},
            "detailed": []
        }
        
        # Find best in each category
        best_cost = min(metrics_list, key=lambda m: m.total_estimated_cost_usd)
        best_speed = max(metrics_list, key=lambda m: m.samples_per_second)
        best_efficiency = min(metrics_list, key=lambda m: m.cost_per_sample_usd)
        
        comparison["summary"] = {
            "best_cost": {"value": best_cost.total_estimated_cost_usd, "config": best_cost.model_name},
            "best_speed": {"value": best_speed.samples_per_second, "config": best_speed.model_name},
            "best_efficiency": {"value": best_efficiency.cost_per_sample_usd, "config": best_efficiency.model_name}
        }
        
        # Detailed comparison
        for i, metrics in enumerate(metrics_list):
            comparison["detailed"].append({
                "label": labels[i],
                "total_cost": metrics.total_estimated_cost_usd,
                "cost_per_sample": metrics.cost_per_sample_usd,
                "samples_per_second": metrics.samples_per_second,
                "total_runtime": metrics.total_runtime,
                "batch_size": metrics.batch_size,
                "quantization": metrics.quantization
            })
        
        return comparison
    
    @staticmethod
    def print_comparison(comparison: Dict):
        """Print formatted comparison results."""
        print("\n" + "="*80)
        print("📊 COST COMPARISON ANALYSIS")
        print("="*80)
        
        summary = comparison.get("summary", {})
        if summary:
            print(f"\n🏆 BEST PERFORMERS:")
            print(f"  Lowest Cost: ${summary['best_cost']['value']:.4f}")
            print(f"  Fastest: {summary['best_speed']['value']:.2f} samples/sec")
            print(f"  Most Efficient: ${summary['best_efficiency']['value']:.6f}/sample")
        
        detailed = comparison.get("detailed", [])
        if detailed:
            print(f"\n📋 DETAILED COMPARISON:")
            print(f"{'Run':<15} {'Total Cost':<12} {'Cost/Sample':<15} {'Speed':<12} {'Runtime':<10} {'Batch':<6} {'Quant':<8}")
            print("-" * 85)
            
            for run in detailed:
                print(f"{run['label']:<15} ${run['total_cost']:<11.4f} ${run['cost_per_sample']:<14.6f} "
                      f"{run['samples_per_second']:<11.2f} {run['total_runtime']:<9.1f}s "
                      f"{run['batch_size']:<6} {run['quantization']:<8}")

if __name__ == "__main__":
    # Demo usage
    tracker = CostTracker("meta-llama/Llama-3.1-8B-Instruct")
    
    tracker.start_run()
    time.sleep(2)  # Simulate some work
    tracker.add_token_count(100, 20)
    tracker.add_sample_processed()
    tracker.end_run()
    
    tracker.print_cost_summary()