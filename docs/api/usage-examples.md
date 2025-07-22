# Usage Examples

## Basic Usage

### Simple Text Generation

```python
from nano_qwen3_serving import LLM, SamplingParams

# Initialize LLM
llm = LLM(model_name="Qwen/Qwen3-0.6B")

# Generate text
result = llm.generate_single(
    "Explain machine learning in simple terms",
    sampling_params=SamplingParams.balanced(max_tokens=200)
)

print(result["generated_text"])
llm.shutdown()
```

### Batch Generation

```python
prompts = [
    "What is artificial intelligence?",
    "How does neural network work?",
    "Explain deep learning"
]

results = llm.generate(
    prompts,
    sampling_params=SamplingParams(temperature=0.7, max_tokens=100)
)

for i, result in enumerate(results):
    print(f"Prompt {i+1}: {result['generated_text']}")
```

### Streaming Generation

```python
print("AI Response: ", end="")
for chunk in llm.generate_stream(
    "Write a short story about a robot",
    sampling_params=SamplingParams.creative(max_tokens=300)
):
    print(chunk["token"], end="", flush=True)
    if chunk["finished"]:
        print("\n\nGeneration complete!")
        break
```

### Chat Interface

```python
conversation = [
    {"role": "system", "content": "You are a helpful programming assistant."},
    {"role": "user", "content": "How do I create a list in Python?"}
]

response = llm.chat(
    conversation,
    sampling_params=SamplingParams(temperature=0.3, max_tokens=150)
)

print("Assistant:", response["generated_text"])
```

### Context Manager Usage

```python
with LLM() as llm:
    # Multiple operations
    result1 = llm.generate_greedy("Define AI")
    result2 = llm.generate_creative("Write a poem about coding")
    
    # Statistics
    stats = llm.get_stats()
    print(f"Total tokens generated: {stats['total_tokens_generated']}")
# Automatic cleanup
```

## Async Usage

### Basic Async Generation

```python
import asyncio
from nano_qwen3_serving import AsyncLLM, SamplingParams

async def main():
    async with AsyncLLM() as llm:
        result = await llm.generate(
            "Explain quantum computing",
            sampling_params=SamplingParams.balanced(max_tokens=200)
        )
        print(result["generated_text"])

asyncio.run(main())
```

### Concurrent Requests

```python
async def generate_multiple():
    async with AsyncLLM() as llm:
        # Submit multiple requests concurrently
        tasks = [
            llm.generate("What is machine learning?"),
            llm.generate("How does AI work?"),
            llm.generate("Explain neural networks")
        ]
        
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            print(f"Response {i+1}: {result['generated_text']}")

asyncio.run(generate_multiple())
```

### Async Streaming

```python
async def stream_chat():
    async with AsyncLLM() as llm:
        messages = [{"role": "user", "content": "Tell me about space exploration"}]
        
        print("AI: ", end="")
        async for chunk in llm.chat_stream(messages):
            print(chunk["token"], end="", flush=True)

asyncio.run(stream_chat())
```

## Custom Sampling Strategies

### Task-Specific Parameters

```python
# Technical documentation
technical_params = SamplingParams(
    temperature=0.2,
    top_p=0.9,
    max_tokens=500,
    repetition_penalty=1.1,
    stop_sequences=["```", "---"]
)

# Creative writing
creative_params = SamplingParams(
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    max_tokens=1000,
    length_penalty=0.8
)

# Code generation
code_params = SamplingParams(
    temperature=0.1,
    top_p=0.95,
    max_tokens=800,
    stop_sequences=["```", "# End", "//END"]
)

# Usage
doc_result = llm.generate_single(
    "Explain how TCP/IP works", 
    sampling_params=technical_params
)

story_result = llm.generate_single(
    "Write a fantasy story about a magic forest",
    sampling_params=creative_params
)

code_result = llm.generate_single(
    "Write a Python function to sort a list",
    sampling_params=code_params
)
```

## Error Handling

### Robust Generation with Retries

```python
from nano_qwen3_serving import LLM, SamplingParams
import logging

def robust_generation(prompt, max_retries=3):
    """Generate text with error handling and retries."""
    
    for attempt in range(max_retries):
        try:
            with LLM() as llm:
                result = llm.generate_single(
                    prompt,
                    sampling_params=SamplingParams.balanced(max_tokens=200)
                )
                return result["generated_text"]
                
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            
    return None

# Usage
try:
    response = robust_generation("Explain quantum computing")
    print(response)
except Exception as e:
    print(f"Generation failed: {e}")
```

## Performance Monitoring

### Benchmarking Generation Performance

```python
import time
from nano_qwen3_serving import LLM

def benchmark_generation():
    """Benchmark generation performance."""
    
    with LLM() as llm:
        # Warm up
        llm.generate_single("Hello", SamplingParams.greedy(max_tokens=10))
        
        # Clear stats for accurate measurement
        llm.clear_stats()
        
        prompts = [
            "Explain machine learning",
            "What is artificial intelligence?",
            "How do neural networks work?",
            "Define deep learning",
            "What is natural language processing?"
        ]
        
        start_time = time.time()
        
        for prompt in prompts:
            result = llm.generate_single(
                prompt,
                SamplingParams(temperature=0.7, max_tokens=100)
            )
            
        end_time = time.time()
        
        # Get final stats
        stats = llm.get_stats()
        
        print(f"Total time: {end_time - start_time:.2f}s")
        print(f"Requests: {len(prompts)}")
        print(f"Tokens generated: {stats.get('total_tokens_generated', 0)}")
        print(f"Average time per request: {(end_time - start_time) / len(prompts):.2f}s")
        print(f"Tokens per second: {stats.get('total_tokens_generated', 0) / (end_time - start_time):.2f}")

benchmark_generation()
```
