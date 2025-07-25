# 调用关系图

## 🔄 请求处理流程

### 1. **HTTP API 请求流程**

```
Client Request
    │
    ▼
┌─────────────────┐
│  FastAPI Server │  ← /v1/chat/completions
│  (openai_server)│
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ OpenAI Service  │  ← OpenAICompatibleService
│ (openai_service)│
└─────────────────┘
    │
    ▼
┌─────────────────┐
│      LLM        │  ← generate_single()
│   (core/llm)    │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   LLMEngine     │  ← generate()
│  (core/engine)  │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  ModelRunner    │  ← run_model()
│ (core/model_runner)│
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ DeviceManager   │  ← optimize_for_device()
│(core/device_manager)│
└─────────────────┘
```

### 2. **异步请求流程**

```
Async Client Request
    │
    ▼
┌─────────────────┐
│   AsyncLLM      │  ← generate()
│ (async/async_llm)│
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ AsyncLLMEngine  │  ← generate_async()
│(async/async_engine)│
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   LLMEngine     │  ← generate() (sync)
│  (core/engine)  │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  ModelRunner    │  ← run_model()
│ (core/model_runner)│
└─────────────────┘
```

## 📋 详细调用关系

### **API Layer 调用关系**

#### FastAPI Server → OpenAI Service
```python
# openai_server.py
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    from .openai_service import service
    return await service.chat_completions(request)
```

#### OpenAI Service → LLM
```python
# openai_service.py
async def chat_completions(self, request: ChatCompletionRequest):
    result = await loop.run_in_executor(
        None,
        lambda: self.llm.generate_single(prompt, sampling_params)
    )
```

### **Core Layer 调用关系**

#### LLM → LLMEngine
```python
# core/llm.py
def generate(self, prompts, sampling_params=None, priority=RequestPriority.NORMAL):
    results = self.engine.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        priority=priority
    )
```

#### LLMEngine → ModelRunner
```python
# core/engine.py
def _process_request(self, request_id: int):
    # Get input tokens
    input_tokens = self.model_runner.tokenize(prompt)
    
    # Run model inference
    outputs = self.model_runner.run_model(
        input_ids=input_tokens,
        past_key_values=past_key_values
    )
```

### **Execution Layer 调用关系**

#### ModelRunner → DeviceManager
```python
# core/model_runner.py
def __init__(self, model_name, device="auto", dtype=None):
    self.device_manager = DeviceManager(device)
    self.device = self.device_manager.device
    
    # Apply device-specific optimizations
    if self.model is not None:
        self.model = self.device_manager.optimize_for_device(self.model)
```

#### LLMEngine → Scheduler
```python
# core/engine.py
def generate(self, prompts, sampling_params=None, priority=RequestPriority.NORMAL):
    # Add requests to scheduler
    request_ids = []
    for prompt in prompts:
        request_id = self.scheduler.add_request(prompt, sampling_params, priority)
        request_ids.append(request_id)
```

#### LLMEngine → BlockManager
```python
# core/engine.py
def _process_request(self, request_id: int):
    # Allocate memory blocks
    block_indices = self.block_manager.allocate_blocks(sequence_length)
    
    # Free blocks when done
    self.block_manager.free_blocks(block_indices)
```

## 🎯 关键接口

### **1. 同步接口**
```python
# 高级接口
llm = LLM(model_name="Qwen/Qwen3-0.6B", device="auto")
result = llm.generate("Hello, world!")

# 引擎接口
engine = LLMEngine(model_name="Qwen/Qwen3-0.6B", device="auto")
results = engine.generate(["Hello", "World"])

# 模型接口
runner = ModelRunner(model_name="Qwen/Qwen3-0.6B", device="auto")
outputs = runner.run_model(input_tokens)
```

### **2. 异步接口**
```python
# 异步高级接口
async with AsyncLLM() as llm:
    result = await llm.generate("Hello, world!")

# 异步引擎接口
async_engine = AsyncLLMEngine()
results = await async_engine.generate_async(["Hello", "World"])
```

### **3. HTTP API接口**
```python
# OpenAI兼容接口
POST /v1/chat/completions
{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
}
```

## 🔧 组件依赖关系

### **初始化依赖**
```
LLM
├── LLMEngine
│   ├── ModelRunner
│   │   └── DeviceManager
│   ├── Scheduler
│   └── BlockManager
└── SamplingParams

AsyncLLM
└── AsyncLLMEngine
    └── LLMEngine (同上)

OpenAICompatibleService
└── LLM (同上)
```

### **运行时依赖**
```
Request Flow:
FastAPI → OpenAI Service → LLM → LLMEngine → ModelRunner → DeviceManager

Async Flow:
AsyncLLM → AsyncLLMEngine → LLMEngine → ModelRunner → DeviceManager

Memory Flow:
LLMEngine → BlockManager → Memory Allocation
LLMEngine → Scheduler → Request Queue
```

## 📊 性能关键路径

### **推理路径**
1. **Tokenization**: `ModelRunner.tokenize()`
2. **Model Forward**: `ModelRunner.run_model()`
3. **Sampling**: `ModelRunner._sample_token()`
4. **Detokenization**: `ModelRunner.detokenize()`

### **批处理路径**
1. **Request Collection**: `Scheduler.get_ready_requests()`
2. **Batch Formation**: `LLMEngine._process_batch()`
3. **Batch Inference**: `ModelRunner.run_model_batch()`
4. **Result Distribution**: `LLMEngine._distribute_results()`

### **内存管理路径**
1. **Block Allocation**: `BlockManager.allocate_blocks()`
2. **KV Cache Management**: `ModelRunner` with `past_key_values`
3. **Block Deallocation**: `BlockManager.free_blocks()`

## 🛡️ 错误处理流程

### **设备错误**
```
DeviceManager.detect_device() → 回退到CPU
ModelRunner._load_model() → 异常处理
```

### **内存错误**
```
BlockManager.allocate_blocks() → 内存不足处理
LLMEngine._process_request() → 请求失败处理
```

### **模型错误**
```
ModelRunner.run_model() → 推理异常处理
LLMEngine.generate() → 生成失败处理
```

## 🔄 并发处理

### **异步并发**
- **AsyncLLM**: 多个并发请求
- **AsyncLLMEngine**: 异步任务队列
- **Worker Pool**: 后台处理线程

### **批处理并发**
- **Scheduler**: 请求队列管理
- **LLMEngine**: 批处理协调
- **ModelRunner**: 批量推理

### **内存并发**
- **BlockManager**: 线程安全的内存分配
- **KV Cache**: 并发访问控制
- **Device Management**: 设备状态同步 