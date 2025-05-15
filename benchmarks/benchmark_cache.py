#!/usr/bin/env python3
"""
Benchmark script for LlamaDiskCache performance and resource usage.
This script can be extended later to compare with Redis semantic cache.
"""

import os
import sys
import time
import json
import logging
import argparse
import psutil
import statistics
from typing import Dict, List, Any, Optional, Tuple

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_cpp import Llama
from llama_cpp.llama_cache import LlamaDiskCache
from distiller_cm5_python.utils.logger import setup_logging

# Setup logging
logger = logging.getLogger(__name__)
setup_logging(log_level=logging.INFO)

class CacheBenchmark:
    """Benchmark class for cache performance testing"""
    
    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        n_ctx: int = 2048,
        capacity_bytes: int = 2 << 30,  # 2GB default
        temperature: float = 0.0,
        num_samples: int = 5,
        repeat_count: int = 3,
    ):
        """Initialize benchmark parameters
        
        Args:
            model_path: Path to the GGUF model file
            cache_dir: Directory to store cache files
            n_ctx: Context size for model
            capacity_bytes: Cache capacity in bytes
            temperature: Sampling temperature
            num_samples: Number of test prompts to generate
            repeat_count: Number of times to repeat each test for averaging
        """
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.n_ctx = n_ctx
        self.capacity_bytes = capacity_bytes
        self.temperature = temperature
        self.num_samples = num_samples
        self.repeat_count = repeat_count
        
        # Results storage
        self.results = {
            "cache_type": "LlamaDiskCache",
            "model_name": os.path.basename(model_path),
            "tests": [],
            "summary": {},
        }
        
        # Make sure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialization metrics
        self.initial_disk_usage = self._get_directory_size(cache_dir)
        self.initial_io_counters = psutil.disk_io_counters()
        self.initial_cpu_times = psutil.cpu_times()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the model and setup cache"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_io_counters = psutil.disk_io_counters()
        start_cpu_times = psutil.cpu_times()
        
        # Load model
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            verbose=False
        )
        
        # Setup cache
        self.model_name = os.path.basename(self.model_path)
        model_cache_dir = os.path.join(self.cache_dir, self.model_name)
        os.makedirs(model_cache_dir, exist_ok=True)
        
        self.cache = LlamaDiskCache(cache_dir=model_cache_dir)
        self.model.set_cache(self.cache)
        
        # Record setup metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        end_io_counters = psutil.disk_io_counters()
        end_cpu_times = psutil.cpu_times()
        
        self.results["setup_time"] = end_time - start_time
        self.results["setup_memory_delta"] = end_memory - start_memory
        self.results["setup_io_delta"] = {
            "read_bytes": end_io_counters.read_bytes - start_io_counters.read_bytes,
            "write_bytes": end_io_counters.write_bytes - start_io_counters.write_bytes,
            "read_count": end_io_counters.read_count - start_io_counters.read_count,
            "write_count": end_io_counters.write_count - start_io_counters.write_count,
        }
        self.results["setup_cpu_delta"] = {
            "user": end_cpu_times.user - start_cpu_times.user,
            "system": end_cpu_times.system - start_cpu_times.system,
            "idle": end_cpu_times.idle - start_cpu_times.idle,
        }
    
    def _get_directory_size(self, path: str) -> int:
        """Calculate total size of a directory in bytes"""
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size
    
    def _get_memory_pressure(self) -> Dict[str, float]:
        """Get memory pressure metrics"""
        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "percent": mem.percent,
            "free": mem.free,
        }
    
    def _get_io_pressure(self) -> Dict[str, float]:
        """Get IO pressure metrics"""
        io_counters = psutil.disk_io_counters()
        io_stats = {
            "read_bytes": io_counters.read_bytes,
            "write_bytes": io_counters.write_bytes,
            "read_count": io_counters.read_count,
            "write_count": io_counters.write_count,
        }
        
        # Add disk usage for mounted filesystems
        partitions = psutil.disk_partitions(all=False)
        for partition in partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                io_stats[f"usage_{partition.mountpoint.replace('/', '_')}"] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent,
                }
            except PermissionError:
                # Skip if we don't have access
                continue
                
        return io_stats
    
    def _get_cpu_cycles(self) -> Dict[str, float]:
        """Get CPU cycles and utilization metrics"""
        cpu_times = psutil.cpu_times()
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        process = psutil.Process()
        
        return {
            "cpu_times": {
                "user": cpu_times.user,
                "system": cpu_times.system,
                "idle": cpu_times.idle,
            },
            "cpu_percent": cpu_percent,
            "process_cpu_percent": process.cpu_percent(interval=0.1),
            "process_cpu_times": {
                "user": process.cpu_times().user,
                "system": process.cpu_times().system,
            },
        }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        process = psutil.Process()
        return {
            "cpu_percent": process.cpu_percent(),
            "memory_usage": process.memory_info().rss,
            "disk_usage": self._get_directory_size(self.cache_dir),
            "memory_pressure": self._get_memory_pressure(),
            "io_pressure": self._get_io_pressure(),
            "cpu_cycles": self._get_cpu_cycles(),
        }
    
    def _generate_test_prompts(self) -> List[str]:
        """Generate a variety of test prompts
        
        Creates prompts with varying degrees of similarity to test 
        both exact and semantic caching effectiveness.
        """
        base_prompts = [
            "Explain the concept of artificial intelligence in simple terms.",
            "What are the main components of a computer?",
            "How does the Internet work?",
            "Describe the process of photosynthesis.",
            "Write a basic function to calculate the factorial of a number."
        ]
        
        # Create variations for testing semantic caching
        all_prompts = []
        for base in base_prompts[:self.num_samples]:
            all_prompts.append(base)
            # Add a semantically similar variant
            all_prompts.append(base.replace("the", "a").replace("is", "are"))
            # Add a variant with different wording but same meaning
            if "explain" in base.lower():
                all_prompts.append(base.lower().replace("explain", "describe"))
        
        return all_prompts[:self.num_samples * 2]  # Limit to requested number of samples
    
    def _format_prompt(self, text: str) -> str:
        """Format prompt for the model"""
        return f"""<|system|>
You are a helpful assistant.
<|user|>
{text}
<|assistant|>"""
    
    def _run_inference(self, prompt: str) -> Tuple[Dict[str, Any], bool]:
        """Run inference with a prompt and measure performance
        
        Returns:
            Tuple of (metrics_dict, is_cache_hit)
        """
        formatted_prompt = self._format_prompt(prompt)
        tokens = self.model.tokenize(formatted_prompt.encode("utf-8"))
        
        # Check if this is a cache hit before inference
        is_cache_hit = False
        try:
            if tokens in self.cache:
                is_cache_hit = True
        except:
            is_cache_hit = False
        
        # Measure start metrics
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        # Run inference
        self.model.reset()
        response = self.model(
            formatted_prompt,
            max_tokens=20,
            temperature=self.temperature,
            echo=False,
        )
        
        # Measure end metrics
        end_time = time.time()
        end_metrics = self._get_system_metrics()
        
        # Calculate deltas
        metrics = {
            "latency": end_time - start_time,
            "memory_delta": end_metrics["memory_usage"] - start_metrics["memory_usage"],
            "disk_delta": end_metrics["disk_usage"] - start_metrics["disk_usage"],
            "cpu_percent": start_metrics["cpu_percent"],
            "is_cache_hit": is_cache_hit,
            "memory_pressure_delta": {
                "used": end_metrics["memory_pressure"]["used"] - start_metrics["memory_pressure"]["used"],
                "percent": end_metrics["memory_pressure"]["percent"] - start_metrics["memory_pressure"]["percent"],
            },
            "io_pressure_delta": {
                "read_bytes": end_metrics["io_pressure"]["read_bytes"] - start_metrics["io_pressure"]["read_bytes"],
                "write_bytes": end_metrics["io_pressure"]["write_bytes"] - start_metrics["io_pressure"]["write_bytes"],
                "read_count": end_metrics["io_pressure"]["read_count"] - start_metrics["io_pressure"]["read_count"],
                "write_count": end_metrics["io_pressure"]["write_count"] - start_metrics["io_pressure"]["write_count"],
            },
            "cpu_cycles_delta": {
                "user": end_metrics["cpu_cycles"]["cpu_times"]["user"] - start_metrics["cpu_cycles"]["cpu_times"]["user"],
                "system": end_metrics["cpu_cycles"]["cpu_times"]["system"] - start_metrics["cpu_cycles"]["cpu_times"]["system"],
                "process_user": end_metrics["cpu_cycles"]["process_cpu_times"]["user"] - start_metrics["cpu_cycles"]["process_cpu_times"]["user"],
                "process_system": end_metrics["cpu_cycles"]["process_cpu_times"]["system"] - start_metrics["cpu_cycles"]["process_cpu_times"]["system"],
            },
        }
        
        return metrics, is_cache_hit
    
    def run_benchmarks(self):
        """Run all benchmark tests"""
        prompts = self._generate_test_prompts()
        logger.info(f"Running benchmarks with {len(prompts)} prompts, {self.repeat_count} iterations each")
        
        # Metrics to track
        all_latencies = {"cache_hit": [], "cache_miss": []}
        all_memory_deltas = {"cache_hit": [], "cache_miss": []}
        all_disk_deltas = {"cache_hit": [], "cache_miss": []}
        all_memory_pressure_deltas = {"cache_hit": [], "cache_miss": []}
        all_io_pressure_deltas = {"cache_hit": [], "cache_miss": []}
        all_cpu_cycles_deltas = {"cache_hit": [], "cache_miss": []}
        cache_hits = 0
        total_runs = 0
        
        # First run: Cache building pass
        logger.info("Initial cache building pass...")
        for i, prompt in enumerate(prompts):
            logger.info(f"  Prompt {i+1}/{len(prompts)}")
            metrics, _ = self._run_inference(prompt)
            # First run is always a cache miss, don't include in stats
        
        # Test runs with multiple repetitions
        logger.info("Running benchmark tests...")
        for iteration in range(self.repeat_count):
            logger.info(f"Iteration {iteration+1}/{self.repeat_count}")
            
            for i, prompt in enumerate(prompts):
                logger.info(f"  Prompt {i+1}/{len(prompts)}")
                
                metrics, is_cache_hit = self._run_inference(prompt)
                
                # Record metrics by cache hit/miss type
                category = "cache_hit" if is_cache_hit else "cache_miss"
                all_latencies[category].append(metrics["latency"])
                all_memory_deltas[category].append(metrics["memory_delta"])
                all_disk_deltas[category].append(metrics["disk_delta"])
                all_memory_pressure_deltas[category].append(metrics["memory_pressure_delta"]["used"])
                all_io_pressure_deltas[category].append(metrics["io_pressure_delta"]["write_bytes"])
                all_cpu_cycles_deltas[category].append(metrics["cpu_cycles_delta"]["process_user"])
                
                if is_cache_hit:
                    cache_hits += 1
                total_runs += 1
                
                # Add individual test result
                self.results["tests"].append({
                    "prompt": prompt,
                    "iteration": iteration,
                    "metrics": metrics
                })
        
        # Calculate summary statistics
        self.results["summary"] = {
            "cache_hit_ratio": cache_hits / total_runs if total_runs > 0 else 0,
            "latency": {
                "cache_hit": {
                    "mean": statistics.mean(all_latencies["cache_hit"]) if all_latencies["cache_hit"] else None,
                    "min": min(all_latencies["cache_hit"]) if all_latencies["cache_hit"] else None,
                    "max": max(all_latencies["cache_hit"]) if all_latencies["cache_hit"] else None,
                },
                "cache_miss": {
                    "mean": statistics.mean(all_latencies["cache_miss"]) if all_latencies["cache_miss"] else None,
                    "min": min(all_latencies["cache_miss"]) if all_latencies["cache_miss"] else None,
                    "max": max(all_latencies["cache_miss"]) if all_latencies["cache_miss"] else None,
                }
            },
            "memory_usage": {
                "cache_hit": {
                    "mean": statistics.mean(all_memory_deltas["cache_hit"]) if all_memory_deltas["cache_hit"] else None,
                },
                "cache_miss": {
                    "mean": statistics.mean(all_memory_deltas["cache_miss"]) if all_memory_deltas["cache_miss"] else None,
                }
            },
            "disk_usage": {
                "cache_hit": {
                    "mean": statistics.mean(all_disk_deltas["cache_hit"]) if all_disk_deltas["cache_hit"] else None,
                },
                "cache_miss": {
                    "mean": statistics.mean(all_disk_deltas["cache_miss"]) if all_disk_deltas["cache_miss"] else None,
                }
            },
            "memory_pressure": {
                "cache_hit": {
                    "mean": statistics.mean(all_memory_pressure_deltas["cache_hit"]) if all_memory_pressure_deltas["cache_hit"] else None,
                },
                "cache_miss": {
                    "mean": statistics.mean(all_memory_pressure_deltas["cache_miss"]) if all_memory_pressure_deltas["cache_miss"] else None,
                }
            },
            "io_pressure": {
                "cache_hit": {
                    "mean": statistics.mean(all_io_pressure_deltas["cache_hit"]) if all_io_pressure_deltas["cache_hit"] else None,
                },
                "cache_miss": {
                    "mean": statistics.mean(all_io_pressure_deltas["cache_miss"]) if all_io_pressure_deltas["cache_miss"] else None,
                }
            },
            "cpu_cycles": {
                "cache_hit": {
                    "mean": statistics.mean(all_cpu_cycles_deltas["cache_hit"]) if all_cpu_cycles_deltas["cache_hit"] else None,
                },
                "cache_miss": {
                    "mean": statistics.mean(all_cpu_cycles_deltas["cache_miss"]) if all_cpu_cycles_deltas["cache_miss"] else None,
                }
            },
            "final_cache_size": self._get_directory_size(self.cache_dir),
            "cache_growth": self._get_directory_size(self.cache_dir) - self.initial_disk_usage,
        }
        
        return self.results
    
    def save_results(self, output_file: str):
        """Save benchmark results to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark LlamaDiskCache performance")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Directory for cache storage")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--samples", type=int, default=5, help="Number of test prompts")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for each test")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context size for model")
    parser.add_argument("--capacity", type=int, default=2 << 30, help="Cache capacity in bytes")
    
    args = parser.parse_args()
    
    benchmark = CacheBenchmark(
        model_path=args.model,
        cache_dir=args.cache_dir,
        n_ctx=args.n_ctx,
        capacity_bytes=args.capacity,
        num_samples=args.samples,
        repeat_count=args.iterations,
    )
    
    results = benchmark.run_benchmarks()
    benchmark.save_results(args.output)
    
    # Print key summary statistics
    print("\nBenchmark Summary:")
    print(f"Cache Hit Ratio: {results['summary']['cache_hit_ratio']:.2%}")
    
    if results['summary']['latency']['cache_hit']['mean'] is not None:
        print(f"Average Cache Hit Latency: {results['summary']['latency']['cache_hit']['mean']:.4f}s")
    
    if results['summary']['latency']['cache_miss']['mean'] is not None:
        print(f"Average Cache Miss Latency: {results['summary']['latency']['cache_miss']['mean']:.4f}s")
    
    print(f"Final Cache Size: {results['summary']['final_cache_size'] / (1024*1024):.2f} MB")
    
    # Print resource pressure stats
    print("\nResource Usage Summary:")
    if results['summary']['memory_pressure']['cache_hit']['mean'] is not None:
        print(f"Memory Pressure (Cache Hit): {results['summary']['memory_pressure']['cache_hit']['mean'] / (1024*1024):.2f} MB")
    
    if results['summary']['io_pressure']['cache_hit']['mean'] is not None:
        print(f"IO Pressure (Cache Hit): {results['summary']['io_pressure']['cache_hit']['mean'] / (1024*1024):.2f} MB")
    
    if results['summary']['cpu_cycles']['cache_hit']['mean'] is not None:
        print(f"CPU Cycles (Cache Hit): {results['summary']['cpu_cycles']['cache_hit']['mean']:.4f}s")


if __name__ == "__main__":
    main() 