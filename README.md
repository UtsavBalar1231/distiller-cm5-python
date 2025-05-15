# Distiller CM5 Python Framework

This repository contains the Python implementation for the Distiller CM5 project, featuring a client-server architecture based on the Model Context Protocal (MCP) for interacting with Large Language Models (LLMs).

## Overview

The framework consists of several key components:

*   **Client (`distiller_cm5_python/client/`)**: Provides user interfaces (both GUI and CLI) to interact with an LLM backend via an MCP server. Includes features like optional local LLM server management, tool usage facilitation, streaming responses, voice input, and E-Ink display support. (See `client/README.md` for details)
*   **LLM Server (`distiller_cm5_python/llm_server/`)**: A FastAPI server that wraps local LLMs (using `llama-cpp-python`) and exposes an OpenAI-compatible API endpoint for chat completions, model management, and caching. (See `llm_server/README.md` for details)
*   **MCP Server (`distiller_cm5_python/mcp_server/`)**: Contains example MCP server implementations that expose specific functionalities (like Text-to-Speech or WiFi control) as tools callable by the MCP client via the protocol. (See `mcp_server/README.md` for details)
*   **Utilities (`distiller_cm5_python/utils/`)**: Shared modules for configuration management (`config.py`), logging (`logger.py`), and custom exceptions (`distiller_exception.py`). (See `utils/README.md` for details)
*   **SDK (`distiller_cm5_sdk`)**: An external, installable package containing reusable components like Whisper (ASR) and Piper (TTS) wrappers. Used by components like the client UI (`client/ui/App.py`) and the talk server (`mcp_server/talk_server.py`). See [Pamir-AI/distiller-cm5-sdk](https://github.com/Pamir-AI/distiller-cm5-sdk/tree/main) for details.
*   **Benchmarks (`benchmarks/`)**: Contains tools for benchmarking and comparing different caching implementations, measuring performance and resource usage metrics like memory pressure, I/O pressure, and CPU cycles. (See `benchmarks/README.md` for details)

The primary user entry point is `main.py` in the project root.

## Redis Semantic Cache

The LLM server now features an optional Redis-based semantic cache that provides:

- Vector similarity search for semantic matching of prompts with similar meanings
- Improved cache hit rates compared to traditional exact-match caching
- Resource management with configurable memory limits, TTL, and similarity thresholds

To use the Redis semantic cache:

1. Install Redis server on your system:
   ```bash
   # For Debian/Ubuntu
   sudo apt install redis-server
   
   # For Arch Linux
   sudo pacman -S redis
   
   # For macOS
   brew install redis
   ```

2. Start the Redis server:
   ```bash
   redis-server
   ```

3. Configure the LLM server to use Redis cache via the `/setCacheType` endpoint:
   ```bash
   curl -X POST "http://localhost:8000/setCacheType" \
     -H "Content-Type: application/json" \
     -d '{
       "type": "redis",
       "redis_host": "localhost",
       "redis_port": 6379,
       "redis_db": 0,
       "similarity_threshold": 0.85,
       "capacity_mb": 2048,
       "ttl": 86400
     }'
   ```

You can also use the benchmark scripts in the `benchmarks/` directory to evaluate the performance of different caching strategies for your specific use case. See `benchmarks/README.md` for detailed instructions on running and analyzing cache performance benchmarks.

## Installation and Setup

1.  **Install `uv` (Recommended Virtual Environment and Package Manager):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env # Or add uv to your PATH manually
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Navigate to the project root directory
    cd /path/to/distiller-cm5-python
    uv venv # Creates a .venv directory
    source .venv/bin/activate # Or `.\.venv\Scripts\activate` on Windows
    ```

3.  **Install Dependencies:**
    ```bash
    # Install this project's requirements AND the external SDK
    uv pip install -r requirements.txt # Ensure you have a requirements.txt file
    uv pip install distiller-cm5-sdk   # Or install from a specific source if needed

    # Example manual installs if requirements.txt is missing:
    # uv pip install fastapi uvicorn "llama-cpp-python[server]" jinja2 pydantic PyQt6 QScintilla PyOpenGL mcp nest_asyncio ...
    # uv pip install distiller-cm5-sdk
    ```
    *Note: Installing `llama-cpp-python` might require C++ build tools. Refer to its documentation.*
    *Note: Installing `PyQt6` might have specific system dependencies.*
    *Note: `distiller-cm5-sdk` might have its own dependencies (like `portaudio` for `PyAudio`) - refer to its [README](https://github.com/Pamir-AI/distiller-cm5-sdk/tree/main).*

4.  **Download LLM Models (for Local LLM Server):**
    Place your desired GGUF-format models into the `distiller_cm5_python/llm_server/models/` directory.

5.  **Download SDK Models/Executables (if needed):**
    Refer to the [distiller-cm5-sdk README](https://github.com/Pamir-AI/distiller-cm5-sdk/tree/main) for instructions on downloading necessary model files (e.g., for Whisper and Piper) or executables required by the SDK components.

6.  **Configuration:**
    *   Copy `distiller_cm5_python/utils/default_config.json` to `mcp_config.json` in the project root (or your working directory) and customize settings like `PROVIDER_TYPE`, `SERVER_URL`, `MODEL_NAME`, etc.
    *   Alternatively, set environment variables (e.g., `export SERVER_URL=http://...`). Configuration loaded by `utils/config.py` prioritizes environment variables, then `mcp_config.json`, then `default_config.json`.

## Running the Application

### 1. LLM Backend

You need an LLM backend accessible. Choose one:

*   **Option A: Run the Local LLM Server:**
    ```bash
    # Make sure environment is activated
    python -m distiller_cm5_python.llm_server.server --model-name your_model.gguf
    ```
    Ensure the `SERVER_URL` in your configuration points to this server (e.g., `http://127.0.0.1:8000`).

*   **Option B: Use OpenRouter or other external API:**
    Set `PROVIDER_TYPE` to `openrouter` (or similar identifier configured in `client/mid_layer/llm_client.py`) in your configuration. Ensure `OPENROUTER_API_KEY` (or equivalent environment variable) is set. The `SERVER_URL` should point to the OpenRouter API endpoint (e.g., `https://openrouter.ai/api/v1`). OpenRouter provides access to various models including OpenAI, Anthropic, Google, etc.

*   **Option C: Use Llama.cpp directly (Managed by Client):**
    Set `PROVIDER_TYPE` to `llama-cpp` in your configuration. The `main.py` script will attempt to start/stop a `llama.cpp` server automatically if one isn't detected at the configured `SERVER_URL`. This requires `llama.cpp` to be compiled and the path to its `server` executable might need configuration (check `client/llm_infra/llama_manager.py` and potentially config).

### 2. MCP Tool Server (Optional)

If you want the client to use tools provided by an MCP server (like TTS or WiFi):

```bash
# Example: Run the talk server (uses distiller_cm5_sdk for TTS)
python -m distiller_cm5_python.mcp_server.talk_server
# Or the wifi server
# python -m distiller_cm5_python.mcp_server.wifi_server
```
Note the path to the *running server script* will be needed when launching the client (`--server-script` argument).

### 3. Client (UI or CLI)

The main entry point handles launching the client.

*   **Run the GUI Client:**
    ```bash
    python main.py --gui [--server-script /path/to/running/mcp_server.py] [OTHER_OPTIONS]
    ```
    *   The `--gui` flag is essential.
    *   Uses `distiller_cm5_sdk` for features like voice input.
    *   Use `--server-script` if you want to connect to a running MCP tool server (like `talk_server.py`).

*   **Run the CLI Client:**
    ```bash
    python main.py [--server-script /path/to/running/mcp_server.py] [OTHER_OPTIONS]
    ```
    *   Omit the `--gui` flag.
    *   Use `--server-script` if needed.

*   **Optional Llama.cpp Management:** If `PROVIDER_TYPE` is `llama-cpp`, `main.py` will check the `SERVER_URL` before starting the client. If no server is running, it will attempt to launch one (requires `llama.cpp` server binary available and configured). It will also attempt to shut down this managed server on exit.

Refer to `python main.py --help` for all client launch options.

