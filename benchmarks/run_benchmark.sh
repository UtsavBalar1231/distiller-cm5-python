#!/bin/bash
# Script to run cache benchmarks and generate comparison report

# Configuration
MODEL_PATH="../distiller_cm5_python/llm_server/models/qwen3-1.7b-q4_k_m.gguf"
LLAMA_CACHE_DIR="./llama_cache_benchmark"
REDIS_CACHE_DIR="./redis_cache_benchmark"
COMPARISON_DIR="./cache_comparison"
SAMPLES=5
ITERATIONS=3
N_CTX=2048

# Create directories
mkdir -p $LLAMA_CACHE_DIR
mkdir -p $REDIS_CACHE_DIR
mkdir -p $COMPARISON_DIR

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
	echo "Error: Model file not found at $MODEL_PATH"
	echo "Please specify the correct path to your model file"
	exit 1
fi

echo "===== LLamaDiskCache Benchmark ====="
python ./benchmark_cache.py \
	--model "$MODEL_PATH" \
	--cache-dir "$LLAMA_CACHE_DIR" \
	--output "$LLAMA_CACHE_DIR/results.json" \
	--samples $SAMPLES \
	--iterations $ITERATIONS \
	--n-ctx $N_CTX

echo ""
echo "===== Redis Cache Benchmark ====="
echo "Checking if Redis is running..."
if ! redis-cli ping &>/dev/null; then
	echo "Redis server not running. Starting Redis server..."
	# Try to start Redis server in the background
	redis-server --daemonize yes --port 6389 --maxmemory 2gb --maxmemory-policy allkeys-lru
	if ! redis-cli -p 6389 ping &>/dev/null; then
		echo "Failed to start Redis server. Please start it manually."
		exit 1
	fi
	echo "Redis server started on port 6389"
	REDIS_PORT=6389
else
	echo "Redis server is already running"
	REDIS_PORT=6379
fi

python ./redis_cache_benchmark.py \
	--model "$MODEL_PATH" \
	--redis-host "localhost" \
	--redis-port $REDIS_PORT \
	--output "$REDIS_CACHE_DIR/results.json" \
	--samples $SAMPLES \
	--iterations $ITERATIONS \
	--n-ctx $N_CTX \
	--capacity-mb 2048 \
	--similarity-threshold 0.85

echo ""
echo "===== Generating Comparison Report ====="
python ./compare_cache_benchmarks.py \
	--llama-results "$LLAMA_CACHE_DIR/results.json" \
	--redis-results "$REDIS_CACHE_DIR/results.json" \
	--output-dir "$COMPARISON_DIR"

echo ""
echo "===== Summary ====="
echo "LlamaDiskCache benchmark results: $LLAMA_CACHE_DIR/results.json"
echo "Redis benchmark results: $REDIS_CACHE_DIR/results.json"
echo "Comparison report: $COMPARISON_DIR/comparison_report.html"
echo ""
echo "To view the full comparison report, open the HTML file in your browser:"
echo "xdg-open $COMPARISON_DIR/comparison_report.html"
