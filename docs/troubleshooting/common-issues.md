# Common Issues and Solutions

This guide covers the most common issues you might encounter when using Nano Qwen3 Serving and how to resolve them.

## 🚨 Critical Issues

### 1. Model Loading Error

**Error:**
```
HFValidationError: Repo id must use alphanumeric chars or '-', '_', '.', '--' and '..' are forbidden, '-' and '.' cannot start or end the name, max length is 96: '<nano_qwen3_serving.core.llm.LLM object at 0x152f8b170>'.
```

**Cause:** The model identifier is being passed as an LLM object instead of a string.

**Solution:**
```python
# ❌ Incorrect - passing LLM object
model_runner = ModelRunner(llm_object)

# ✅ Correct - passing string identifier
model_runner = ModelRunner("Qwen/Qwen3-0.6B")
```

**Fix in code:**
```python
# In nano_qwen3_serving/core/model_runner.py
def __init__(self, model_name: str, device: str = "mps"):
    self.model_name = model_name  # Should be string like "Qwen/Qwen3-0.6B"
    self.device = device
    self._load_model()

def _load_model(self):
    # Use self.model_name (string) instead of passing LLM object
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
```

### 2. 404 Errors for API Endpoints

**Error:**
```
INFO: 127.0.0.1:49260 - "GET /v1/models HTTP/1.1" 404 Not Found
INFO: 127.0.0.1:49279 - "POST /v1/chat/completions HTTP/1.1" 404 Not Found
```

**Cause:** The API routes are not properly registered or the server is not running the correct application.

**Solution:**
1. Check that the FastAPI app includes all required routes
2. Ensure the server is running the correct application instance
3. Verify the route paths match OpenAI API specification

**Fix:**
```python
# In nano_qwen3_serving/service/server.py
app = FastAPI(title="Nano Qwen3 Serving", version="1.0.0")

# Add all required routes
app.add_api_route("/health", health_check, methods=["GET"])
app.add_api_route("/v1/models", list_models, methods=["GET"])
app.add_api_route("/v1/chat/completions", chat_completions, methods=["POST"])
app.add_api_route("/stats", get_stats, methods=["GET"])
```

## 🔧 Common Issues

### 3. Out of Memory (OOM)

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Use a smaller model:**
   ```bash
   python -m nano_qwen3_serving --model Qwen/Qwen3-0.6B
   ```

2. **Reduce batch size:**
   ```python
   # In configuration
   max_batch_size = 1
   ```

3. **Use CPU instead of MPS:**
   ```bash
   python -m nano_qwen3_serving --device cpu
   ```

4. **Increase system memory or use swap:**
   ```bash
   # Check available memory
   vm_stat
   
   # Create swap file if needed
   sudo sysctl vm.swapusage
   ```

### 4. MPS Not Available

**Error:**
```
RuntimeError: MPS not available
```

**Solutions:**
1. **Check macOS version:** Ensure you're on macOS 12.3+ (Monterey)
2. **Check Apple Silicon:** Verify you have M1/M2/M3 chip
3. **Update PyTorch:** Install latest PyTorch with MPS support
   ```bash
   pip install torch torchvision torchaudio
   ```

4. **Fallback to CPU:**
   ```bash
   python -m nano_qwen3_serving --device cpu
   ```

### 5. Model Download Issues

**Error:**
```
ConnectionError: Failed to download model
```

**Solutions:**
1. **Check internet connection**
2. **Use Hugging Face token:**
   ```bash
   export HUGGING_FACE_HUB_TOKEN=your_token
   ```

3. **Download manually:**
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   
   model_name = "Qwen/Qwen3-0.6B"
   tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")
   model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./models")
   ```

4. **Use local model path:**
   ```bash
   python -m nano_qwen3_serving --model ./local/path/to/model
   ```

### 6. Port Already in Use

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solutions:**
1. **Find and kill the process:**
   ```bash
   lsof -ti:8000 | xargs kill -9
   ```

2. **Use a different port:**
   ```bash
   python -m nano_qwen3_serving --port 8001
   ```

3. **Check what's using the port:**
   ```bash
   lsof -i :8000
   ```

### 7. Slow Performance

**Symptoms:**
- High response times
- Low tokens per second
- High memory usage

**Solutions:**
1. **Optimize model settings:**
   ```python
   # Reduce context length
   max_context_length = 512
   
   # Use smaller batch size
   max_batch_size = 1
   ```

2. **Enable optimizations:**
   ```python
   # Use torch.compile (PyTorch 2.0+)
   model = torch.compile(model)
   
   # Use half precision
   model = model.half()
   ```

3. **Monitor system resources:**
   ```bash
   # Check CPU usage
   top
   
   # Check memory usage
   vm_stat
   
   # Check GPU usage (if available)
   sudo powermetrics --samplers gpu_power -n 1
   ```

### 8. Pydantic Warnings

**Warning:**
```
UserWarning: Field "model_info" has conflict with protected namespace "model_".
```

**Solution:**
```python
# In your Pydantic model
class Config:
    protected_namespaces = ()
```

## 🔍 Debugging Techniques

### 1. Enable Debug Logging

```bash
python -m nano_qwen3_serving --log-level debug
```

### 2. Check Server Status

```bash
curl http://localhost:8000/health
```

### 3. Monitor Logs

```bash
# Follow logs in real-time
tail -f logs/nano_qwen3.log

# Search for errors
grep -i error logs/nano_qwen3.log
```

### 4. Test Individual Components

```python
# Test model loading
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Test inference
inputs = tokenizer("Hello", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## 📊 Performance Monitoring

### 1. Check Performance Stats

```bash
curl http://localhost:8000/stats
```

### 2. Monitor System Resources

```bash
# CPU and memory
htop

# GPU usage (if available)
sudo powermetrics --samplers gpu_power -n 1

# Network connections
netstat -an | grep 8000
```

### 3. Benchmark Performance

```python
import time
import requests

def benchmark_api():
    start_time = time.time()
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50
        }
    )
    end_time = time.time()
    
    print(f"Response time: {end_time - start_time:.3f}s")
    print(f"Status code: {response.status_code}")
    return response.json()

# Run benchmark
result = benchmark_api()
```

## 🆘 Getting Help

### 1. Check Logs

Always check the logs first:
```bash
tail -n 100 logs/nano_qwen3.log
```

### 2. Search Issues

Check existing issues on GitHub:
- [GitHub Issues](https://github.com/hsliuustc/nano-qwen3-serving/issues)
- [GitHub Discussions](https://github.com/hsliuustc/nano-qwen3-serving/discussions)

### 3. Create Issue

When creating an issue, include:
- Error message and stack trace
- System information (macOS version, Python version)
- Model and configuration used
- Steps to reproduce
- Logs (if applicable)

### 4. Community Support

- Join our [Discussions](https://github.com/hsliuustc/nano-qwen3-serving/discussions)
- Check the [FAQ](../user-guide/faq.md)
- Review [Performance Guide](performance.md)

---

**💡 Pro Tip:** Most issues can be resolved by checking the logs and ensuring you're using the latest version of the package. 