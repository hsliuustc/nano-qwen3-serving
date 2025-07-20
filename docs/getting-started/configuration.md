# Configuration Guide

This guide covers all configuration options for Nano Qwen3 Serving, from basic settings to advanced tuning.

## 🔧 Basic Configuration

### Command Line Options

```bash
python -m nano_qwen3_serving [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--port` | int | `8000` | Server port |
| `--host` | str | `127.0.0.1` | Server host |
| `--model` | str | `Qwen/Qwen3-0.6B` | Model to load |
| `--device` | str | `mps` | Device (mps/cpu) |
| `--workers` | int | `1` | Number of workers |
| `--log-level` | str | `info` | Logging level |
| `--max-context-length` | int | `2048` | Maximum context length |
| `--max-batch-size` | int | `1` | Maximum batch size |

### Environment Variables

Set these environment variables for persistent configuration:

```bash
# Server settings
export NANO_QWEN3_PORT=8000
export NANO_QWEN3_HOST=127.0.0.1
export NANO_QWEN3_MODEL=Qwen/Qwen3-0.6B
export NANO_QWEN3_DEVICE=mps

# Performance settings
export NANO_QWEN3_MAX_CONTEXT_LENGTH=2048
export NANO_QWEN3_MAX_BATCH_SIZE=1
export NANO_QWEN3_WORKERS=1

# Logging
export NANO_QWEN3_LOG_LEVEL=info

# Hugging Face
export HUGGING_FACE_HUB_TOKEN=your_token
export HF_HOME=./models
```

## 📁 Configuration File

Create a `config.yaml` file for advanced configuration:

```yaml
# Server Configuration
server:
  host: "127.0.0.1"
  port: 8000
  workers: 1
  log_level: "info"

# Model Configuration
model:
  name: "Qwen/Qwen3-0.6B"
  device: "mps"
  max_context_length: 2048
  max_batch_size: 1
  torch_dtype: "float16"  # float16, float32, bfloat16

# Performance Configuration
performance:
  enable_compilation: true
  enable_attention_slicing: false
  enable_gradient_checkpointing: false
  memory_efficient_attention: true

# Caching Configuration
cache:
  model_cache_dir: "./models"
  enable_disk_cache: true
  max_cache_size: "10GB"

# Logging Configuration
logging:
  level: "info"
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
  file: "logs/nano_qwen3.log"
  rotation: "10 MB"
  retention: "7 days"

# API Configuration
api:
  enable_cors: true
  cors_origins: ["*"]
  rate_limit_enabled: false
  rate_limit_requests: 100
  rate_limit_window: 60
```

## 🎛️ Advanced Configuration

### Model-Specific Settings

```yaml
models:
  qwen3-0.6b:
    name: "Qwen/Qwen3-0.6B"
    device: "mps"
    max_context_length: 2048
    torch_dtype: "float16"
    
  qwen3-1.5b:
    name: "Qwen/Qwen3-1.5B"
    device: "mps"
    max_context_length: 4096
    torch_dtype: "float16"
    
  qwen3-3b:
    name: "Qwen/Qwen3-3B"
    device: "mps"
    max_context_length: 8192
    torch_dtype: "float16"
```

### Performance Optimization

```yaml
performance:
  # Model compilation (PyTorch 2.0+)
  enable_compilation: true
  compilation_mode: "reduce-overhead"  # default, reduce-overhead, max-autotune
  
  # Memory optimization
  enable_attention_slicing: false
  attention_slice_size: 1
  
  # Gradient checkpointing (for training)
  enable_gradient_checkpointing: false
  
  # Memory efficient attention
  memory_efficient_attention: true
  
  # Flash attention (if available)
  enable_flash_attention: false
  
  # Quantization
  quantization:
    enabled: false
    method: "int8"  # int8, int4, fp4
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Memory Management

```yaml
memory:
  # GPU memory management
  max_gpu_memory: "80%"  # Percentage or absolute value like "8GB"
  gpu_memory_fraction: 0.8
  
  # CPU memory management
  max_cpu_memory: "16GB"
  
  # Cache management
  enable_memory_pool: true
  memory_pool_size: "2GB"
  
  # Garbage collection
  gc_threshold: 0.8  # Trigger GC when memory usage > 80%
  gc_interval: 100   # GC every 100 requests
```

## 🔒 Security Configuration

### Authentication

```yaml
security:
  # API key authentication
  enable_auth: false
  api_keys:
    - "sk-your-api-key-here"
    - "sk-another-key"
  
  # Rate limiting
  rate_limit:
    enabled: true
    requests_per_minute: 60
    burst_size: 10
    
  # CORS settings
  cors:
    enabled: true
    origins: ["http://localhost:3000", "https://yourdomain.com"]
    methods: ["GET", "POST", "OPTIONS"]
    headers: ["Content-Type", "Authorization"]
    
  # Request validation
  validation:
    max_request_size: "10MB"
    max_tokens_per_request: 4096
    allowed_models: ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.5B"]
```

### Network Security

```yaml
network:
  # SSL/TLS configuration
  ssl:
    enabled: false
    cert_file: "certs/server.crt"
    key_file: "certs/server.key"
    
  # Proxy configuration
  proxy:
    enabled: false
    upstream: "http://proxy.example.com:8080"
    
  # Firewall rules
  firewall:
    allowed_ips: ["127.0.0.1", "192.168.1.0/24"]
    blocked_ips: []
```

## 📊 Monitoring Configuration

### Metrics and Monitoring

```yaml
monitoring:
  # Prometheus metrics
  prometheus:
    enabled: true
    port: 9090
    path: "/metrics"
    
  # Health checks
  health:
    enabled: true
    interval: 30  # seconds
    timeout: 5    # seconds
    
  # Performance monitoring
  performance:
    enabled: true
    metrics_interval: 60  # seconds
    enable_detailed_stats: true
    
  # Logging
  logging:
    enable_request_logging: true
    enable_performance_logging: true
    log_sensitive_data: false
```

### Alerting

```yaml
alerts:
  # Memory alerts
  memory:
    warning_threshold: 0.8  # 80%
    critical_threshold: 0.95  # 95%
    
  # Response time alerts
  response_time:
    warning_threshold: 5.0  # seconds
    critical_threshold: 10.0  # seconds
    
  # Error rate alerts
  error_rate:
    warning_threshold: 0.05  # 5%
    critical_threshold: 0.1  # 10%
```

## 🔄 Deployment Configuration

### Production Settings

```yaml
production:
  # Server settings
  server:
    host: "0.0.0.0"  # Listen on all interfaces
    port: 8000
    workers: 4  # Number of CPU cores
    
  # Model settings
  model:
    name: "Qwen/Qwen3-1.5B"
    device: "mps"
    max_context_length: 4096
    
  # Performance settings
  performance:
    enable_compilation: true
    memory_efficient_attention: true
    
  # Security settings
  security:
    enable_auth: true
    rate_limit:
      enabled: true
      requests_per_minute: 100
      
  # Monitoring
  monitoring:
    prometheus:
      enabled: true
    health:
      enabled: true
```

### Docker Configuration

```yaml
docker:
  # Container settings
  container:
    name: "nano-qwen3-serving"
    image: "hsliuustc/nano-qwen3-serving:latest"
    
  # Volume mounts
  volumes:
    - "./models:/app/models"
    - "./logs:/app/logs"
    - "./config:/app/config"
    
  # Environment variables
  environment:
    NANO_QWEN3_MODEL: "Qwen/Qwen3-0.6B"
    NANO_QWEN3_DEVICE: "mps"
    NANO_QWEN3_LOG_LEVEL: "info"
    
  # Port mapping
  ports:
    - "8000:8000"
    - "9090:9090"  # Prometheus metrics
```

## 🧪 Development Configuration

### Development Settings

```yaml
development:
  # Server settings
  server:
    host: "127.0.0.1"
    port: 8000
    workers: 1
    
  # Model settings
  model:
    name: "Qwen/Qwen3-0.6B"
    device: "mps"
    max_context_length: 1024
    
  # Debug settings
  debug:
    enabled: true
    log_level: "debug"
    enable_profiling: true
    
  # Testing
  testing:
    enable_mock_responses: false
    test_model: "Qwen/Qwen3-0.6B"
```

## 📝 Configuration Examples

### Minimal Configuration

```bash
# Start with minimal settings
python -m nano_qwen3_serving --port 8000 --model Qwen/Qwen3-0.6B
```

### High-Performance Configuration

```bash
# Optimized for performance
python -m nano_qwen3_serving \
  --port 8000 \
  --model Qwen/Qwen3-1.5B \
  --device mps \
  --max-context-length 4096 \
  --max-batch-size 4 \
  --workers 2
```

### Development Configuration

```bash
# Development settings
python -m nano_qwen3_serving \
  --port 8001 \
  --model Qwen/Qwen3-0.6B \
  --log-level debug \
  --max-context-length 1024
```

### Production Configuration

```bash
# Production settings
export NANO_QWEN3_HOST=0.0.0.0
export NANO_QWEN3_PORT=8000
export NANO_QWEN3_MODEL=Qwen/Qwen3-1.5B
export NANO_QWEN3_WORKERS=4
export NANO_QWEN3_LOG_LEVEL=warning

python -m nano_qwen3_serving
```

## 🔍 Configuration Validation

### Validate Configuration

```python
import yaml
from nano_qwen3_serving.config import validate_config

def validate_config_file(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        validate_config(config)
        print("Configuration is valid!")
        return True
    except Exception as e:
        print(f"Configuration error: {e}")
        return False

# Validate your config
validate_config_file("config.yaml")
```

### Configuration Schema

```yaml
# Configuration schema (for reference)
schema:
  server:
    host: str
    port: int
    workers: int
    log_level: str
    
  model:
    name: str
    device: str
    max_context_length: int
    max_batch_size: int
    torch_dtype: str
    
  performance:
    enable_compilation: bool
    enable_attention_slicing: bool
    memory_efficient_attention: bool
```

## 📚 Next Steps

- **[Quick Start](quick-start.md)**: Get up and running quickly
- **[Installation](installation.md)**: Installation guide
- **[API Reference](../user-guide/api-reference.md)**: Complete API documentation
- **[Troubleshooting](../troubleshooting/common-issues.md)**: Common issues and solutions

---

**💡 Tip:** Start with minimal configuration and gradually add features as needed. Monitor performance and adjust settings accordingly. 