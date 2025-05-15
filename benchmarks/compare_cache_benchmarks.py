#!/usr/bin/env python3
"""
Compare benchmark results between LlamaDiskCache and Redis semantic cache.
"""

import os
import sys
import json
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distiller_cm5_python.utils.logger import setup_logging

# Setup logging
logger = logging.getLogger(__name__)
setup_logging(log_level=logging.INFO)

class CacheComparer:
    """Compare cache benchmark results"""
    
    def __init__(
        self,
        llama_disk_cache_results: str,
        redis_cache_results: str,
        output_dir: str = "./comparison_results",
    ):
        """Initialize with paths to result files
        
        Args:
            llama_disk_cache_results: Path to LlamaDiskCache benchmark results
            redis_cache_results: Path to Redis cache benchmark results
            output_dir: Directory to save comparison results and plots
        """
        self.llama_disk_cache_results_path = llama_disk_cache_results
        self.redis_cache_results_path = redis_cache_results
        self.output_dir = output_dir
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load benchmark results
        self.llama_results = self._load_results(llama_disk_cache_results)
        self.redis_results = self._load_results(redis_cache_results)
        
        # Comparative results
        self.comparison = {
            "llama_disk_cache": {
                "name": "LlamaDiskCache",
                "summary": self.llama_results.get("summary", {}),
            },
            "redis_cache": {
                "name": "Redis Semantic Cache",
                "summary": self.redis_results.get("summary", {}),
            },
            "comparative_metrics": {},
        }
    
    def _load_results(self, file_path: str) -> Dict[str, Any]:
        """Load benchmark results from a JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results from {file_path}: {e}")
            return {}
    
    def calculate_comparative_metrics(self) -> Dict[str, Any]:
        """Calculate comparative metrics between the two cache types"""
        metrics = {}
        
        # Return empty metrics if either result set is missing critical data
        if not self.llama_results.get("summary") or not self.redis_results.get("summary"):
            logger.warning("One or both result sets are missing summary data")
            if "note" in self.redis_results.get("summary", {}):
                metrics["note"] = "Redis implementation is a placeholder"
            return metrics
        
        # Cache hit ratio comparison
        llama_hit_ratio = self.llama_results["summary"].get("cache_hit_ratio", 0)
        redis_hit_ratio = self.redis_results["summary"].get("cache_hit_ratio", 0)
        
        metrics["cache_hit_ratio"] = {
            "llama_disk_cache": llama_hit_ratio,
            "redis_cache": redis_hit_ratio,
            "delta": redis_hit_ratio - llama_hit_ratio,
            "delta_percent": (
                ((redis_hit_ratio / llama_hit_ratio) - 1) * 100
                if llama_hit_ratio > 0 else float("inf")
            ),
        }
        
        # Latency comparison for cache hits
        llama_hit_latency = self.llama_results["summary"].get("latency", {}).get("cache_hit", {}).get("mean")
        redis_hit_latency = self.redis_results["summary"].get("latency", {}).get("cache_hit", {}).get("mean")
        
        if llama_hit_latency is not None and redis_hit_latency is not None:
            metrics["hit_latency"] = {
                "llama_disk_cache": llama_hit_latency,
                "redis_cache": redis_hit_latency,
                "delta": redis_hit_latency - llama_hit_latency,
                "delta_percent": (
                    ((redis_hit_latency / llama_hit_latency) - 1) * 100
                    if llama_hit_latency > 0 else float("inf")
                ),
            }
        
        # Latency comparison for cache misses
        llama_miss_latency = self.llama_results["summary"].get("latency", {}).get("cache_miss", {}).get("mean")
        redis_miss_latency = self.redis_results["summary"].get("latency", {}).get("cache_miss", {}).get("mean")
        
        if llama_miss_latency is not None and redis_miss_latency is not None:
            metrics["miss_latency"] = {
                "llama_disk_cache": llama_miss_latency,
                "redis_cache": redis_miss_latency,
                "delta": redis_miss_latency - llama_miss_latency,
                "delta_percent": (
                    ((redis_miss_latency / llama_miss_latency) - 1) * 100
                    if llama_miss_latency > 0 else float("inf")
                ),
            }
        
        # Memory usage comparison
        llama_cache_size = self.llama_results["summary"].get("final_cache_size", 0)
        redis_memory = self.redis_results["summary"].get("final_redis_memory", 0)
        
        metrics["memory_usage"] = {
            "llama_disk_cache": llama_cache_size,
            "redis_cache": redis_memory,
            "delta": redis_memory - llama_cache_size,
            "delta_percent": (
                ((redis_memory / llama_cache_size) - 1) * 100
                if llama_cache_size > 0 else float("inf")
            ),
        }
        
        # Memory pressure comparison
        llama_memory_pressure = self.llama_results["summary"].get("memory_pressure", {}).get("cache_hit", {}).get("mean")
        redis_memory_pressure = self.redis_results["summary"].get("memory_pressure", {}).get("cache_hit", {}).get("mean")
        
        if llama_memory_pressure is not None and redis_memory_pressure is not None:
            metrics["memory_pressure"] = {
                "llama_disk_cache": llama_memory_pressure,
                "redis_cache": redis_memory_pressure,
                "delta": redis_memory_pressure - llama_memory_pressure,
                "delta_percent": (
                    ((redis_memory_pressure / llama_memory_pressure) - 1) * 100
                    if llama_memory_pressure > 0 else float("inf")
                ),
            }
        
        # IO pressure comparison
        llama_io_pressure = self.llama_results["summary"].get("io_pressure", {}).get("cache_hit", {}).get("mean")
        redis_io_pressure = self.redis_results["summary"].get("io_pressure", {}).get("cache_hit", {}).get("mean")
        
        if llama_io_pressure is not None and redis_io_pressure is not None:
            metrics["io_pressure"] = {
                "llama_disk_cache": llama_io_pressure,
                "redis_cache": redis_io_pressure,
                "delta": redis_io_pressure - llama_io_pressure,
                "delta_percent": (
                    ((redis_io_pressure / llama_io_pressure) - 1) * 100
                    if llama_io_pressure > 0 else float("inf")
                ),
            }
        
        # CPU cycles comparison
        llama_cpu_cycles = self.llama_results["summary"].get("cpu_cycles", {}).get("cache_hit", {}).get("mean")
        redis_cpu_cycles = self.redis_results["summary"].get("cpu_cycles", {}).get("cache_hit", {}).get("mean")
        
        if llama_cpu_cycles is not None and redis_cpu_cycles is not None:
            metrics["cpu_cycles"] = {
                "llama_disk_cache": llama_cpu_cycles,
                "redis_cache": redis_cpu_cycles,
                "delta": redis_cpu_cycles - llama_cpu_cycles,
                "delta_percent": (
                    ((redis_cpu_cycles / llama_cpu_cycles) - 1) * 100
                    if llama_cpu_cycles > 0 else float("inf")
                ),
            }
        
        return metrics
    
    def generate_plots(self):
        """Generate comparative plots"""
        # Skip plotting if Redis is just a placeholder
        if "note" in self.redis_results.get("summary", {}):
            logger.info("Skipping plots as Redis implementation is a placeholder")
            return
        
        # Create figure with subplots - 3x2 grid to include resource metrics
        fig, axs = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('Cache Performance Comparison', fontsize=16)
        
        # Cache hit ratio
        cache_types = ['LlamaDiskCache', 'Redis Cache']
        hit_ratios = [
            self.llama_results["summary"].get("cache_hit_ratio", 0) * 100,
            self.redis_results["summary"].get("cache_hit_ratio", 0) * 100
        ]
        
        axs[0, 0].bar(cache_types, hit_ratios)
        axs[0, 0].set_title('Cache Hit Ratio (%)')
        axs[0, 0].set_ylim(0, 100)
        
        # Latency comparison
        llama_hit_latency = self.llama_results["summary"].get("latency", {}).get("cache_hit", {}).get("mean", 0)
        llama_miss_latency = self.llama_results["summary"].get("latency", {}).get("cache_miss", {}).get("mean", 0)
        redis_hit_latency = self.redis_results["summary"].get("latency", {}).get("cache_hit", {}).get("mean", 0)
        redis_miss_latency = self.redis_results["summary"].get("latency", {}).get("cache_miss", {}).get("mean", 0)
        
        latencies = [
            [llama_hit_latency or 0, redis_hit_latency or 0],
            [llama_miss_latency or 0, redis_miss_latency or 0]
        ]
        
        x = np.arange(len(cache_types))
        width = 0.35
        
        axs[0, 1].bar(x - width/2, latencies[0], width, label='Cache Hit')
        axs[0, 1].bar(x + width/2, latencies[1], width, label='Cache Miss')
        axs[0, 1].set_xticks(x)
        axs[0, 1].set_xticklabels(cache_types)
        axs[0, 1].set_title('Latency (seconds)')
        axs[0, 1].legend()
        
        # Memory usage
        llama_memory = self.llama_results["summary"].get("final_cache_size", 0) / (1024*1024)  # MB
        redis_memory = self.redis_results["summary"].get("final_redis_memory", 0) / (1024*1024)  # MB
        
        memory = [llama_memory, redis_memory]
        
        axs[1, 0].bar(cache_types, memory)
        axs[1, 0].set_title('Memory Usage (MB)')
        
        # Memory pressure
        llama_memory_pressure = self.llama_results["summary"].get("memory_pressure", {}).get("cache_hit", {}).get("mean", 0) / (1024*1024)  # MB
        redis_memory_pressure = self.redis_results["summary"].get("memory_pressure", {}).get("cache_hit", {}).get("mean", 0) / (1024*1024)  # MB
        
        memory_pressure = [llama_memory_pressure, redis_memory_pressure]
        
        axs[1, 1].bar(cache_types, memory_pressure)
        axs[1, 1].set_title('Memory Pressure (MB)')
        
        # IO pressure
        llama_io_pressure = self.llama_results["summary"].get("io_pressure", {}).get("cache_hit", {}).get("mean", 0) / (1024*1024)  # MB
        redis_io_pressure = self.redis_results["summary"].get("io_pressure", {}).get("cache_hit", {}).get("mean", 0) / (1024*1024)  # MB
        
        io_pressure = [llama_io_pressure, redis_io_pressure]
        
        axs[2, 0].bar(cache_types, io_pressure)
        axs[2, 0].set_title('IO Pressure (MB)')
        
        # CPU cycles
        llama_cpu_cycles = self.llama_results["summary"].get("cpu_cycles", {}).get("cache_hit", {}).get("mean", 0)
        redis_cpu_cycles = self.redis_results["summary"].get("cpu_cycles", {}).get("cache_hit", {}).get("mean", 0)
        
        cpu_cycles = [llama_cpu_cycles, redis_cpu_cycles]
        
        axs[2, 1].bar(cache_types, cpu_cycles)
        axs[2, 1].set_title('CPU Cycles (seconds)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cache_comparison.png'))
        plt.close()
    
    def generate_report(self):
        """Generate a comparative report"""
        # Calculate comparative metrics
        self.comparison["comparative_metrics"] = self.calculate_comparative_metrics()
        
        # Generate plots if possible
        try:
            self.generate_plots()
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
        
        # Save comparison results
        with open(os.path.join(self.output_dir, 'comparison_results.json'), 'w') as f:
            json.dump(self.comparison, f, indent=2)
        
        # Generate HTML report
        self._generate_html_report()
        
        logger.info(f"Comparison report generated in {self.output_dir}")
        return self.comparison
    
    def _generate_html_report(self):
        """Generate an HTML report with the comparison results"""
        # Basic HTML template for the report
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cache Benchmark Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .img-container {{ margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Cache Benchmark Comparison</h1>
            <p>Comparing performance between LlamaDiskCache and Redis semantic cache.</p>
            
            <h2>Cache Hit Ratio</h2>
            <table>
                <tr>
                    <th>Cache Type</th>
                    <th>Hit Ratio</th>
                    <th>Difference</th>
                </tr>
                <tr>
                    <td>LlamaDiskCache</td>
                    <td>{llama_hit_ratio:.2%}</td>
                    <td rowspan="2" class="{hit_ratio_class}">{hit_ratio_diff:.2%}</td>
                </tr>
                <tr>
                    <td>Redis Cache</td>
                    <td>{redis_hit_ratio:.2%}</td>
                </tr>
            </table>
            
            <h2>Latency (seconds)</h2>
            <table>
                <tr>
                    <th>Cache Type</th>
                    <th>Cache Hit</th>
                    <th>Cache Miss</th>
                    <th>Hit/Miss Difference</th>
                </tr>
                <tr>
                    <td>LlamaDiskCache</td>
                    <td>{llama_hit_latency:.4f}</td>
                    <td>{llama_miss_latency:.4f}</td>
                    <td>{llama_latency_diff:.4f}</td>
                </tr>
                <tr>
                    <td>Redis Cache</td>
                    <td>{redis_hit_latency:.4f}</td>
                    <td>{redis_miss_latency:.4f}</td>
                    <td>{redis_latency_diff:.4f}</td>
                </tr>
                <tr>
                    <td>Difference</td>
                    <td class="{hit_latency_class}">{hit_latency_diff:.4f}</td>
                    <td class="{miss_latency_class}">{miss_latency_diff:.4f}</td>
                    <td></td>
                </tr>
            </table>
            
            <h2>Memory Usage</h2>
            <table>
                <tr>
                    <th>Cache Type</th>
                    <th>Size (MB)</th>
                    <th>Difference</th>
                </tr>
                <tr>
                    <td>LlamaDiskCache</td>
                    <td>{llama_memory:.2f}</td>
                    <td rowspan="2" class="{memory_class}">{memory_diff:.2f} MB ({memory_diff_percent:.2f}%)</td>
                </tr>
                <tr>
                    <td>Redis Cache</td>
                    <td>{redis_memory:.2f}</td>
                </tr>
            </table>
            
            <h2>Resource Pressure Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>LlamaDiskCache</th>
                    <th>Redis Cache</th>
                    <th>Difference</th>
                </tr>
                <tr>
                    <td>Memory Pressure (MB)</td>
                    <td>{llama_memory_pressure:.2f}</td>
                    <td>{redis_memory_pressure:.2f}</td>
                    <td class="{memory_pressure_class}">{memory_pressure_diff:.2f} ({memory_pressure_diff_percent:.2f}%)</td>
                </tr>
                <tr>
                    <td>IO Pressure (MB)</td>
                    <td>{llama_io_pressure:.2f}</td>
                    <td>{redis_io_pressure:.2f}</td>
                    <td class="{io_pressure_class}">{io_pressure_diff:.2f} ({io_pressure_diff_percent:.2f}%)</td>
                </tr>
                <tr>
                    <td>CPU Cycles (s)</td>
                    <td>{llama_cpu_cycles:.4f}</td>
                    <td>{redis_cpu_cycles:.4f}</td>
                    <td class="{cpu_cycles_class}">{cpu_cycles_diff:.4f} ({cpu_cycles_diff_percent:.2f}%)</td>
                </tr>
            </table>
            
            <h2>Visualization</h2>
            <div class="img-container">
                <img src="cache_comparison.png" alt="Cache Comparison Chart">
            </div>
            
            <p><em>Generated on {timestamp}</em></p>
        </body>
        </html>
        """
        
        # Helper functions
        def format_percent(value):
            """Format a value as a percentage"""
            if value is None:
                return "N/A"
            return f"{value:.2%}"
        
        def format_percent_change(value):
            """Format a percentage change with sign"""
            if value is None:
                return "N/A"
            return f"{value:+.2f}%"
        
        def format_number(value):
            """Format a number to 4 decimal places"""
            if value is None:
                return "N/A"
            return f"{value:.4f}"
        
        def format_memory(value):
            """Format memory value in MB"""
            if value is None:
                return "N/A"
            return f"{value / (1024*1024):.2f}"
        
        # Extract values for the report
        metrics = self.comparison["comparative_metrics"]
        
        llama_hit_ratio = self.llama_results["summary"].get("cache_hit_ratio", 0)
        redis_hit_ratio = self.redis_results["summary"].get("cache_hit_ratio", 0)
        hit_ratio_diff = redis_hit_ratio - llama_hit_ratio
        hit_ratio_class = "positive" if hit_ratio_diff > 0 else "negative"
        
        llama_hit_latency = self.llama_results["summary"].get("latency", {}).get("cache_hit", {}).get("mean", 0) or 0
        llama_miss_latency = self.llama_results["summary"].get("latency", {}).get("cache_miss", {}).get("mean", 0) or 0
        redis_hit_latency = self.redis_results["summary"].get("latency", {}).get("cache_hit", {}).get("mean", 0) or 0
        redis_miss_latency = self.redis_results["summary"].get("latency", {}).get("cache_miss", {}).get("mean", 0) or 0
        
        llama_latency_diff = llama_miss_latency - llama_hit_latency
        redis_latency_diff = redis_miss_latency - redis_hit_latency
        hit_latency_diff = redis_hit_latency - llama_hit_latency
        miss_latency_diff = redis_miss_latency - llama_miss_latency
        
        hit_latency_class = "negative" if hit_latency_diff > 0 else "positive"
        miss_latency_class = "negative" if miss_latency_diff > 0 else "positive"
        
        llama_memory = self.llama_results["summary"].get("final_cache_size", 0) / (1024*1024)
        redis_memory = self.redis_results["summary"].get("final_redis_memory", 0) / (1024*1024)
        memory_diff = redis_memory - llama_memory
        memory_diff_percent = (memory_diff / llama_memory * 100) if llama_memory > 0 else 0
        memory_class = "negative" if memory_diff > 0 else "positive"
        
        # Resource pressure metrics
        llama_memory_pressure = self.llama_results["summary"].get("memory_pressure", {}).get("cache_hit", {}).get("mean", 0) / (1024*1024) or 0
        redis_memory_pressure = self.redis_results["summary"].get("memory_pressure", {}).get("cache_hit", {}).get("mean", 0) / (1024*1024) or 0
        memory_pressure_diff = redis_memory_pressure - llama_memory_pressure
        memory_pressure_diff_percent = (memory_pressure_diff / llama_memory_pressure * 100) if llama_memory_pressure > 0 else 0
        memory_pressure_class = "negative" if memory_pressure_diff > 0 else "positive"
        
        llama_io_pressure = self.llama_results["summary"].get("io_pressure", {}).get("cache_hit", {}).get("mean", 0) / (1024*1024) or 0
        redis_io_pressure = self.redis_results["summary"].get("io_pressure", {}).get("cache_hit", {}).get("mean", 0) / (1024*1024) or 0
        io_pressure_diff = redis_io_pressure - llama_io_pressure
        io_pressure_diff_percent = (io_pressure_diff / llama_io_pressure * 100) if llama_io_pressure > 0 else 0
        io_pressure_class = "negative" if io_pressure_diff > 0 else "positive"
        
        llama_cpu_cycles = self.llama_results["summary"].get("cpu_cycles", {}).get("cache_hit", {}).get("mean", 0) or 0
        redis_cpu_cycles = self.redis_results["summary"].get("cpu_cycles", {}).get("cache_hit", {}).get("mean", 0) or 0
        cpu_cycles_diff = redis_cpu_cycles - llama_cpu_cycles
        cpu_cycles_diff_percent = (cpu_cycles_diff / llama_cpu_cycles * 100) if llama_cpu_cycles > 0 else 0
        cpu_cycles_class = "negative" if cpu_cycles_diff > 0 else "positive"
        
        # Current timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Fill in the template
        html_content = html_template.format(
            llama_hit_ratio=llama_hit_ratio,
            redis_hit_ratio=redis_hit_ratio,
            hit_ratio_diff=hit_ratio_diff,
            hit_ratio_class=hit_ratio_class,
            
            llama_hit_latency=llama_hit_latency,
            llama_miss_latency=llama_miss_latency,
            redis_hit_latency=redis_hit_latency,
            redis_miss_latency=redis_miss_latency,
            llama_latency_diff=llama_latency_diff,
            redis_latency_diff=redis_latency_diff,
            hit_latency_diff=hit_latency_diff,
            miss_latency_diff=miss_latency_diff,
            hit_latency_class=hit_latency_class,
            miss_latency_class=miss_latency_class,
            
            llama_memory=llama_memory,
            redis_memory=redis_memory,
            memory_diff=memory_diff,
            memory_diff_percent=memory_diff_percent,
            memory_class=memory_class,
            
            llama_memory_pressure=llama_memory_pressure,
            redis_memory_pressure=redis_memory_pressure,
            memory_pressure_diff=memory_pressure_diff,
            memory_pressure_diff_percent=memory_pressure_diff_percent,
            memory_pressure_class=memory_pressure_class,
            
            llama_io_pressure=llama_io_pressure,
            redis_io_pressure=redis_io_pressure,
            io_pressure_diff=io_pressure_diff,
            io_pressure_diff_percent=io_pressure_diff_percent,
            io_pressure_class=io_pressure_class,
            
            llama_cpu_cycles=llama_cpu_cycles,
            redis_cpu_cycles=redis_cpu_cycles,
            cpu_cycles_diff=cpu_cycles_diff,
            cpu_cycles_diff_percent=cpu_cycles_diff_percent,
            cpu_cycles_class=cpu_cycles_class,
            
            timestamp=timestamp
        )
        
        # Write HTML to file
        with open(os.path.join(self.output_dir, 'comparison_report.html'), 'w') as f:
            f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Compare LlamaDiskCache and Redis cache benchmark results")
    parser.add_argument("--llama-results", type=str, required=True, help="Path to LlamaDiskCache benchmark results JSON")
    parser.add_argument("--redis-results", type=str, required=True, help="Path to Redis cache benchmark results JSON")
    parser.add_argument("--output-dir", type=str, default="./cache_comparison", help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    comparer = CacheComparer(
        llama_disk_cache_results=args.llama_results,
        redis_cache_results=args.redis_results,
        output_dir=args.output_dir,
    )
    
    comparison = comparer.generate_report()
    
    # Print key comparative metrics
    print("\nComparative Benchmark Results:")
    
    metrics = comparison["comparative_metrics"]
    
    if "cache_hit_ratio" in metrics:
        llama_hit = metrics["cache_hit_ratio"]["llama_disk_cache"]
        redis_hit = metrics["cache_hit_ratio"]["redis_cache"]
        delta_percent = metrics["cache_hit_ratio"]["delta_percent"]
        print(f"Cache Hit Ratio: LlamaDiskCache {llama_hit:.2%} vs Redis {redis_hit:.2%} ({delta_percent:+.2f}%)")
    
    if "hit_latency" in metrics:
        llama_latency = metrics["hit_latency"]["llama_disk_cache"]
        redis_latency = metrics["hit_latency"]["redis_cache"]
        delta_percent = metrics["hit_latency"]["delta_percent"]
        print(f"Cache Hit Latency: LlamaDiskCache {llama_latency:.4f}s vs Redis {redis_latency:.4f}s ({delta_percent:+.2f}%)")
    
    if "memory_usage" in metrics:
        llama_mem = metrics["memory_usage"]["llama_disk_cache"] / (1024*1024)
        redis_mem = metrics["memory_usage"]["redis_cache"] / (1024*1024)
        delta_percent = metrics["memory_usage"]["delta_percent"]
        print(f"Memory Usage: LlamaDiskCache {llama_mem:.2f}MB vs Redis {redis_mem:.2f}MB ({delta_percent:+.2f}%)")
    
    # Print resource pressure metrics
    if "memory_pressure" in metrics:
        llama_mem_pressure = metrics["memory_pressure"]["llama_disk_cache"] / (1024*1024)
        redis_mem_pressure = metrics["memory_pressure"]["redis_cache"] / (1024*1024)
        delta_percent = metrics["memory_pressure"]["delta_percent"]
        print(f"Memory Pressure: LlamaDiskCache {llama_mem_pressure:.2f}MB vs Redis {redis_mem_pressure:.2f}MB ({delta_percent:+.2f}%)")
    
    if "io_pressure" in metrics:
        llama_io = metrics["io_pressure"]["llama_disk_cache"] / (1024*1024)
        redis_io = metrics["io_pressure"]["redis_cache"] / (1024*1024)
        delta_percent = metrics["io_pressure"]["delta_percent"]
        print(f"IO Pressure: LlamaDiskCache {llama_io:.2f}MB vs Redis {redis_io:.2f}MB ({delta_percent:+.2f}%)")
    
    if "cpu_cycles" in metrics:
        llama_cpu = metrics["cpu_cycles"]["llama_disk_cache"]
        redis_cpu = metrics["cpu_cycles"]["redis_cache"]
        delta_percent = metrics["cpu_cycles"]["delta_percent"]
        print(f"CPU Cycles: LlamaDiskCache {llama_cpu:.4f}s vs Redis {redis_cpu:.4f}s ({delta_percent:+.2f}%)")
    
    print(f"\nDetailed report generated in {args.output_dir}")


if __name__ == "__main__":
    main() 