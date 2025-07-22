# Service Configuration

## Starting the Service

### Command Line

```bash
# Basic start
python tools/start_service.py

# Custom configuration
python tools/start_service.py --host 0.0.0.0 --port 8080 --workers 4

# Development mode
python tools/start_service.py --reload --log-level debug
```

### Command Line Options

```bash
python tools/start_service.py --help
```

**Available Options:**
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--workers`: Number of worker processes
- `--log-level`: Logging level (debug, info, warning, error)
- `--model-name`: Model to load (default: Qwen/Qwen3-0.6B)
- `--device`: Device to use (default: mps)
- `--dtype`: Data type (default: float16)

### Programmatic Start

```python
import uvicorn
from nano_qwen3_serving.service.openai_server import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        workers=1,
        log_level="info"
    )
```

## Environment Variables

### Model Configuration

```bash
export MODEL_NAME="Qwen/Qwen3-0.6B"
export DEVICE="mps"
export DTYPE="float16"
```

### Memory Configuration

```bash
export MAX_QUEUE_SIZE=1000
export NUM_BLOCKS=1024
export BLOCK_SIZE=16
export MAX_SEQ_LENGTH=4096
```

### Server Configuration

```bash
export HOST="127.0.0.1"
export PORT=8000
export WORKERS=1
export LOG_LEVEL="info"
```

## Configuration Examples

### Production Configuration

```bash
#!/bin/bash
# production_start.sh

export MODEL_NAME="Qwen/Qwen3-0.6B"
export DEVICE="mps"
export DTYPE="float16"
export HOST="0.0.0.0"
export PORT=8000
export WORKERS=4
export LOG_LEVEL="warning"
export MAX_QUEUE_SIZE=2000
export NUM_BLOCKS=2048

python tools/start_service.py \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --log-level $LOG_LEVEL
```

### Development Configuration

```bash
#!/bin/bash
# dev_start.sh

export MODEL_NAME="Qwen/Qwen3-0.6B"
export DEVICE="mps"
export DTYPE="float16"
export HOST="127.0.0.1"
export PORT=8000
export LOG_LEVEL="debug"

python tools/start_service.py \
    --host $HOST \
    --port $PORT \
    --reload \
    --log-level $LOG_LEVEL
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

ENV MODEL_NAME="Qwen/Qwen3-0.6B"
ENV DEVICE="cpu"
ENV DTYPE="float16"
ENV HOST="0.0.0.0"
ENV PORT=8000

EXPOSE 8000

CMD ["python", "tools/start_service.py", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  nano-qwen3-serving:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=Qwen/Qwen3-0.6B
      - DEVICE=cpu
      - DTYPE=float16
      - HOST=0.0.0.0
      - PORT=8000
      - WORKERS=1
    volumes:
      - ./models:/app/models
```

## Health Monitoring

### Health Check Script

```python
#!/usr/bin/env python3
"""Health check script for the service."""

import requests
import sys
import time

def check_health(base_url="http://127.0.0.1:8000", timeout=30):
    """Check if the service is healthy."""
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"✅ Service is healthy!")
                print(f"Status: {health['status']}")
                print(f"Uptime: {health['uptime']:.2f} seconds")
                print(f"Model: {health['model_info']['model_name']}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
    
    print("❌ Service is not healthy")
    return False

if __name__ == "__main__":
    healthy = check_health()
    sys.exit(0 if healthy else 1)
```

### Performance Monitoring

```python
#!/usr/bin/env python3
"""Performance monitoring script."""

import requests
import time
import json

def monitor_performance(base_url="http://127.0.0.1:8000", interval=10):
    """Monitor service performance."""
    
    while True:
        try:
            # Get stats
            response = requests.get(f"{base_url}/stats")
            if response.status_code == 200:
                stats = response.json()
                
                print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Requests served: {stats.get('requests_served', 0)}")
                print(f"Tokens generated: {stats.get('total_tokens_generated', 0)}")
                print(f"Avg response time: {stats.get('average_response_time', 0):.2f}s")
                print(f"Queue length: {stats.get('queue_length', 0)}")
                print(f"Active requests: {stats.get('active_requests', 0)}")
                print("-" * 50)
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting stats: {e}")
        
        time.sleep(interval)

if __name__ == "__main__":
    monitor_performance()
```

## Load Testing

### Simple Load Test

```python
#!/usr/bin/env python3
"""Simple load test for the service."""

import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor

async def send_request(session, url, data):
    """Send a single request."""
    try:
        async with session.post(url, json=data) as response:
            result = await response.json()
            return result
    except Exception as e:
        return {"error": str(e)}

async def load_test(base_url="http://127.0.0.1:8000", concurrent_requests=10, total_requests=100):
    """Run a load test."""
    
    url = f"{base_url}/v1/chat/completions"
    data = {
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def limited_request():
            async with semaphore:
                return await send_request(session, url, data)
        
        # Run all requests
        tasks = [limited_request() for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Analyze results
        successful = sum(1 for r in results if "error" not in r)
        failed = total_requests - successful
        
        print(f"Load Test Results:")
        print(f"Total requests: {total_requests}")
        print(f"Concurrent requests: {concurrent_requests}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total time: {end_time - start_time:.2f}s")
        print(f"Requests per second: {total_requests / (end_time - start_time):.2f}")

if __name__ == "__main__":
    asyncio.run(load_test())
```

## Configuration Best Practices

### Memory Management

```python
# For 8GB RAM Apple Silicon
LLM(
    model_name="Qwen/Qwen3-0.6B",
    num_blocks=512,
    block_size=16,
    max_seq_length=2048
)

# For 16GB RAM Apple Silicon
LLM(
    model_name="Qwen/Qwen3-1.5B",
    num_blocks=1024,
    block_size=16,
    max_seq_length=4096
)
```

### Production Deployment

```python
# Production configuration
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    workers=4,                    # Multiple workers for better throughput
    log_level="warning",          # Reduce log verbosity
    access_log=False,            # Disable access logs for performance
    loop="uvloop",               # Use faster event loop (if available)
    http="httptools"             # Use faster HTTP parser
)
```

### Security Configuration

```python
# Add security headers
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com"])
app.add_middleware(HTTPSRedirectMiddleware)  # For HTTPS deployment
```
