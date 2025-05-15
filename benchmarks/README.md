# Cache Benchmarking Tools

This directory contains tools for benchmarking and comparing different caching implementations for LLM inference.

## Overview

The benchmarking suite provides tools to evaluate the performance and resource utilization of different caching strategies:

- **LlamaDiskCache**: A file-based caching system for storing LLM responses
- **Redis Semantic Cache**: A Redis-based caching system with semantic similarity matching

The benchmarks measure:
- Cache hit ratio
- Latency (cache hit vs. miss)
- Memory usage
- Memory pressure
- I/O pressure
- CPU cycles

## Files

- `benchmark_cache.py`: Benchmarks the LlamaDiskCache implementation
- `redis_cache_benchmark.py`: Benchmarks the Redis semantic cache implementation
- `compare_cache_benchmarks.py`: Compares results from both caching implementations
- `run_benchmark.sh`: Shell script to run all benchmarks and generate comparison reports

## Requirements

- Python 3.9+
- Redis server (for Redis semantic cache benchmarks)
- Required Python packages:
  - `llama-cpp-python`
  - `psutil`
  - `redis`
  - `matplotlib`
  - `numpy`

## Usage

### Running the Complete Benchmark Suite

Run all benchmarks and generate a comparison report:

```bash
cd benchmarks
./run_benchmark.sh
```

This will:
1. Run the LlamaDiskCache benchmark
2. Run the Redis semantic cache benchmark
3. Generate a comparison report with charts
4. Create a detailed HTML report

### Running Individual Benchmarks

#### LlamaDiskCache Benchmark

```bash
python benchmark_cache.py \
    --model "path/to/model.gguf" \
    --cache-dir "./cache" \
    --output "llama_results.json" \
    --samples 5 \
    --iterations 3 \
    --n-ctx 2048
```

#### Redis Semantic Cache Benchmark

```bash
python redis_cache_benchmark.py \
    --model "path/to/model.gguf" \
    --redis-host "localhost" \
    --redis-port 6379 \
    --output "redis_results.json" \
    --samples 5 \
    --iterations 3 \
    --n-ctx 2048 \
    --capacity-mb 2048 \
    --similarity-threshold 0.85
```

#### Comparing Results

```bash
python compare_cache_benchmarks.py \
    --llama-results "llama_results.json" \
    --redis-results "redis_results.json" \
    --output-dir "./comparison_results"
```

## Command-Line Arguments

### LlamaDiskCache Benchmark (`benchmark_cache.py`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to the GGUF model file | (Required) |
| `--cache-dir` | Directory for cache storage | `./cache` |
| `--output` | Output file for results | `benchmark_results.json` |
| `--samples` | Number of test prompts | 5 |
| `--iterations` | Number of iterations for each test | 3 |
| `--n-ctx` | Context size for model | 2048 |
| `--capacity` | Cache capacity in bytes | 2GB |

### Redis Semantic Cache Benchmark (`redis_cache_benchmark.py`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to the GGUF model file | (Required) |
| `--redis-host` | Redis server hostname | `localhost` |
| `--redis-port` | Redis server port | 6379 |
| `--redis-db` | Redis database number | 0 |
| `--redis-password` | Redis password | None |
| `--output` | Output file for results | `redis_benchmark_results.json` |
| `--samples` | Number of test prompts | 5 |
| `--iterations` | Number of iterations for each test | 3 |
| `--n-ctx` | Context size for model | 2048 |
| `--similarity-threshold` | Semantic similarity threshold (0-1) | 0.85 |
| `--capacity-mb` | Redis memory capacity in MB | 2048 |
| `--ttl` | Cache TTL in seconds | 3600 |
| `--embedding-model` | Embedding model to use | `all-MiniLM-L6-v2` |

### Comparison Tool (`compare_cache_benchmarks.py`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--llama-results` | Path to LlamaDiskCache benchmark results | (Required) |
| `--redis-results` | Path to Redis cache benchmark results | (Required) |
| `--output-dir` | Directory for comparison results | `./cache_comparison` |

## Metrics Measured

The benchmarks collect and analyze the following metrics:

### Performance Metrics
- **Cache Hit Ratio**: Percentage of requests that are served from cache
- **Latency**: Time to serve requests (comparing cache hits vs. misses)
- **Semantic Caching Effectiveness**: For Redis, measures ability to match semantically similar queries

### Resource Usage Metrics
- **Memory Usage**: RAM used by the caching system
- **Memory Pressure**: System-wide memory pressure during cache operations
- **I/O Pressure**: Disk I/O operations during cache operations 
- **CPU Cycles**: CPU usage during cache operations

## Output

The benchmarks generate:

1. JSON result files with detailed metrics
2. Console output with summary statistics
3. Comparison charts (PNG format)
4. Interactive HTML report with detailed analysis

## Modifying the Benchmarks

The benchmarking code is structured to make it easy to:
- Add new cache implementations
- Extend with additional metrics
- Customize test prompts and evaluation criteria

To add a new cache implementation, create a new benchmark file following the pattern in the existing files. 