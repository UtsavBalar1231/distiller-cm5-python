#!/usr/bin/env python3
"""
Benchmark script for Redis semantic cache performance and resource usage.
This script follows the same structure as benchmark_cache.py to allow fair comparison.
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

# Import Redis semantic cache
try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama_cpp module not found. Please install it with:")
    print("pip install llama-cpp-python")
    sys.exit(1)

import redis
from distiller_cm5_python.llm_server.redis_semantic_cache import RedisSemanticCache
from distiller_cm5_python.utils.logger import setup_logging

# Setup logging
logger = logging.getLogger(__name__)
setup_logging(log_level=logging.INFO)

class RedisSemanticCacheBenchmark:
    """Benchmark class for Redis semantic cache performance testing"""
    
    def __init__(
        self,
        model_path: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        n_ctx: int = 2048,
        cache_ttl: int = 3600,  # Cache TTL in seconds
        similarity_threshold: float = 0.85,  # Threshold for semantic similarity
        capacity_mb: int = 2048,  # 2GB default
        temperature: float = 0.0,
        num_samples: int = 5,
        repeat_count: int = 3,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize benchmark parameters
        
        Args:
            model_path: Path to the GGUF model file
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (if any)
            n_ctx: Context size for model
            cache_ttl: Time-to-live for cache entries in seconds
            similarity_threshold: Threshold for semantic similarity (0-1)
            capacity_mb: Redis memory capacity in MB
            temperature: Sampling temperature
            num_samples: Number of test prompts to generate
            repeat_count: Number of times to repeat each test for averaging
            embedding_model: Sentence transformer model to use for embeddings
        """
        self.model_path = model_path
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        self.n_ctx = n_ctx
        self.cache_ttl = cache_ttl
        self.similarity_threshold = similarity_threshold
        self.capacity_mb = capacity_mb
        self.temperature = temperature
        self.num_samples = num_samples
        self.repeat_count = repeat_count
        self.embedding_model = embedding_model
        
        # Results storage
        self.results = {
            "cache_type": "RedisSemanticCache",
            "model_name": os.path.basename(model_path),
            "tests": [],
            "summary": {},
        }
        
        # Connect to Redis to check if it's available
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                socket_timeout=5,
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
            
            # Record initial memory usage
            self.initial_redis_info = self.redis_client.info("memory")
            self.initial_memory_usage = int(self.initial_redis_info.get("used_memory", 0))
            self.initial_io_counters = psutil.disk_io_counters()
            self.initial_cpu_times = psutil.cpu_times()
            
            # Initialize model and cache
            self._load_model()
            
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.error("Please ensure Redis server is running and accessible")
            raise
        except Exception as e:
            logger.error(f"Error initializing benchmark: {e}")
            raise
    
    def _load_model(self):
        """Load the model and setup cache"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_io_counters = psutil.disk_io_counters()
        start_cpu_times = psutil.cpu_times()
        
        try:
            # Load model
            logger.info(f"Loading model from {self.model_path}")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                verbose=False
            )
            
            # Setup Redis semantic cache
            self.model_name = os.path.basename(self.model_path)
            logger.info(f"Initializing Redis semantic cache for model {self.model_name}")
            
            self.cache = RedisSemanticCache(
                redis_host=self.redis_host,
                redis_port=self.redis_port,
                redis_db=self.redis_db,
                redis_password=self.redis_password,
                model_name=self.model_name,
                embedding_model=self.embedding_model,
                similarity_threshold=self.similarity_threshold,
                ttl=self.cache_ttl,
                capacity_mb=self.capacity_mb,
            )
            
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
            
            # Store Redis metrics
            redis_info = self.redis_client.info("memory")
            self.results["redis_setup"] = {
                "used_memory_bytes": int(redis_info.get("used_memory", 0)),
                "used_memory_human": redis_info.get("used_memory_human", "unknown"),
                "maxmemory_bytes": int(redis_info.get("maxmemory", 0)),
                "maxmemory_human": redis_info.get("maxmemory_human", "unknown"),
                "maxmemory_policy": redis_info.get("maxmemory_policy", "unknown"),
            }
            
            logger.info(f"Model and cache setup complete in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model or setup cache: {e}")
            raise
    
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
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource usage"""
        process = psutil.Process()
        
        # Get Redis memory usage
        redis_info = self.redis_client.info("memory")
        redis_memory = int(redis_info.get("used_memory", 0))
        
        return {
            "cpu_percent": process.cpu_percent(),
            "memory_usage": process.memory_info().rss,
            "redis_memory": redis_memory,
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
            elif "describe" in base.lower():
                all_prompts.append(base.lower().replace("describe", "explain"))
            elif "what" in base.lower():
                all_prompts.append(base.lower().replace("what are", "list"))
            elif "how" in base.lower():
                all_prompts.append(base.lower().replace("how", "in what way"))
        
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
        
        # Measure start metrics
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        # Check for cache hit and run inference
        is_cache_hit = self.cache.load_state(
            self.model, 
            prompt=formatted_prompt, 
            tokens=tokens,
            semantic_search=True
        )
        
        if not is_cache_hit:
            # Cache miss, run inference and save to cache
            self.model.reset()
            response = self.model(
                formatted_prompt,
                max_tokens=20,
                temperature=self.temperature,
                echo=False,
            )
            
            # Save to cache
            self.cache.save_state(self.model, formatted_prompt, tokens)
        
        # Measure end metrics
        end_time = time.time()
        end_metrics = self._get_system_metrics()
        
        # Calculate deltas
        metrics = {
            "latency": end_time - start_time,
            "memory_delta": end_metrics["memory_usage"] - start_metrics["memory_usage"],
            "redis_memory_delta": end_metrics["redis_memory"] - start_metrics["redis_memory"],
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
        all_redis_memory_deltas = {"cache_hit": [], "cache_miss": []}
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
                all_redis_memory_deltas[category].append(metrics["redis_memory_delta"])
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
        
        # Get final Redis memory usage
        redis_info = self.redis_client.info("memory")
        final_redis_memory = int(redis_info.get("used_memory", 0))
        
        # Calculate summary statistics
        self.results["summary"] = {
            "cache_hit_ratio": cache_hits / total_runs if total_runs > 0 else 0,
            "semantic_caching_effectiveness": sum(1 for t in self.results["tests"] if t["metrics"]["is_cache_hit"] and "the" not in t["prompt"]) / \
                                              sum(1 for t in self.results["tests"] if "the" not in t["prompt"]) if sum(1 for t in self.results["tests"] if "the" not in t["prompt"]) > 0 else 0,
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
            "redis_memory_usage": {
                "cache_hit": {
                    "mean": statistics.mean(all_redis_memory_deltas["cache_hit"]) if all_redis_memory_deltas["cache_hit"] else None,
                },
                "cache_miss": {
                    "mean": statistics.mean(all_redis_memory_deltas["cache_miss"]) if all_redis_memory_deltas["cache_miss"] else None,
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
            "initial_redis_memory": self.initial_memory_usage,
            "final_redis_memory": final_redis_memory,
            "redis_memory_growth": final_redis_memory - self.initial_memory_usage,
            "redis_info": {
                "used_memory_human": redis_info.get("used_memory_human", "unknown"),
                "maxmemory_human": redis_info.get("maxmemory_human", "unknown"),
                "maxmemory_policy": redis_info.get("maxmemory_policy", "unknown"),
            }
        }
        
        # Add cache statistics
        cache_stats = self.cache.get_stats()
        self.results["summary"]["cache_entries"] = cache_stats["entries"]
        
        # Clean up Redis cache (optional)
        if self.cache:
            self.cache.clear()
            logger.info("Cleared Redis cache after benchmark")
        
        return self.results
    
    def save_results(self, output_file: str):
        """Save benchmark results to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Redis semantic cache performance")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database number")
    parser.add_argument("--redis-password", type=str, default=None, help="Redis password")
    parser.add_argument("--output", type=str, default="redis_benchmark_results.json", help="Output file for results")
    parser.add_argument("--samples", type=int, default=5, help="Number of test prompts")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for each test")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context size for model")
    parser.add_argument("--similarity-threshold", type=float, default=0.85, help="Semantic similarity threshold (0-1)")
    parser.add_argument("--capacity-mb", type=int, default=2048, help="Redis memory capacity in MB")
    parser.add_argument("--ttl", type=int, default=3600, help="Cache TTL in seconds")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model to use")
    
    args = parser.parse_args()
    
    try:
        # Check if Redis is running before starting benchmark
        r = redis.Redis(
            host=args.redis_host, 
            port=args.redis_port, 
            db=args.redis_db,
            password=args.redis_password,
            socket_timeout=5
        )
        r.ping()
        logger.info("Redis is available, starting benchmark")
        
        benchmark = RedisSemanticCacheBenchmark(
            model_path=args.model,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_db=args.redis_db,
            redis_password=args.redis_password,
            n_ctx=args.n_ctx,
            similarity_threshold=args.similarity_threshold,
            capacity_mb=args.capacity_mb,
            cache_ttl=args.ttl,
            num_samples=args.samples,
            repeat_count=args.iterations,
            embedding_model=args.embedding_model,
        )
        
        results = benchmark.run_benchmarks()
        benchmark.save_results(args.output)
        
        # Print key summary statistics
        print("\nBenchmark Summary:")
        print(f"Cache Hit Ratio: {results['summary']['cache_hit_ratio']:.2%}")
        print(f"Semantic Caching Effectiveness: {results['summary']['semantic_caching_effectiveness']:.2%}")
        
        if results['summary']['latency']['cache_hit']['mean'] is not None:
            print(f"Average Cache Hit Latency: {results['summary']['latency']['cache_hit']['mean']:.4f}s")
        
        if results['summary']['latency']['cache_miss']['mean'] is not None:
            print(f"Average Cache Miss Latency: {results['summary']['latency']['cache_miss']['mean']:.4f}s")
        
        print(f"Final Redis Memory Usage: {results['summary']['redis_info']['used_memory_human']}")
        print(f"Redis Cache Entries: {results['summary']['cache_entries']}")
        
        # Print resource pressure stats
        print("\nResource Usage Summary:")
        if results['summary']['memory_pressure']['cache_hit']['mean'] is not None:
            print(f"Memory Pressure (Cache Hit): {results['summary']['memory_pressure']['cache_hit']['mean'] / (1024*1024):.2f} MB")
        
        if results['summary']['io_pressure']['cache_hit']['mean'] is not None:
            print(f"IO Pressure (Cache Hit): {results['summary']['io_pressure']['cache_hit']['mean'] / (1024*1024):.2f} MB")
        
        if results['summary']['cpu_cycles']['cache_hit']['mean'] is not None:
            print(f"CPU Cycles (Cache Hit): {results['summary']['cpu_cycles']['cache_hit']['mean']:.4f}s")
        
    except redis.exceptions.ConnectionError:
        logger.error("Redis server is not running or not accessible")
        logger.error("Please start Redis server before running this benchmark")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
