# Async Components API Documentation

## AsyncLLM Class

High-level async interface for the nano LLM serving engine.

### Constructor

```python
from nano_qwen3_serving import AsyncLLM

async_llm = AsyncLLM(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str = "mps",
    dtype: str = "float16",
    max_queue_size: int = 1000,
    num_blocks: int = 1024,
    block_size: int = 16,
    max_seq_length: int = 4096,
    worker_count: int = 2
)
```

### Async Methods

#### generate()
Async text generation.

```python
async def example():
    async with AsyncLLM() as llm:
        result = await llm.generate("Hello, async world!")
        print(result["generated_text"])
```

#### generate_stream()
Async streaming generation.

```python
async def stream_example():
    async with AsyncLLM() as llm:
        async for result in llm.generate_stream("Tell me a story"):
            print(result["token"], end="", flush=True)
```

#### chat() and chat_stream()
Async chat methods.

```python
# Async chat
result = await llm.chat(messages)

# Async chat streaming
async for result in llm.chat_stream(messages):
    print(result["token"], end="")
```

#### generate_batch()
Efficient batch processing.

```python
prompts = ["Question 1", "Question 2", "Question 3"]
results = await llm.generate_batch(prompts, batch_size=2)
```

### Request Management

```python
# Submit request and get ID
request_id = await llm.submit_request("Hello")

# Get result later
result = await llm.get_result(request_id)
```

### Async Context Manager

```python
async with AsyncLLM() as llm:
    result = await llm.generate("Hello!")
    # Automatic startup and cleanup
```

## Usage Examples

### Concurrent Requests

```python
async def generate_multiple():
    async with AsyncLLM() as llm:
        tasks = [
            llm.generate("What is machine learning?"),
            llm.generate("How does AI work?"),
            llm.generate("Explain neural networks")
        ]
        
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            print(f"Response {i+1}: {result['generated_text']}")
```

### Batch Processing

```python
async def process_batch():
    prompts = [
        "Summarize machine learning",
        "Explain artificial intelligence", 
        "Define deep learning",
        "What is natural language processing?"
    ]
    
    async with AsyncLLM() as llm:
        results = await llm.generate_batch(
            prompts, 
            batch_size=2,
            sampling_params=SamplingParams(temperature=0.5, max_tokens=100)
        )
        
        for prompt, result in zip(prompts, results):
            print(f"Q: {prompt}")
            print(f"A: {result['generated_text']}\n")
```
