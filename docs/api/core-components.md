# Core Components API Documentation

## LLM Class

The main high-level interface for text generation.

### Constructor

```python
from nano_qwen3_serving import LLM

llm = LLM(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str = "mps",
    dtype: str = "float16",
    max_queue_size: int = 1000,
    num_blocks: int = 1024,
    block_size: int = 16,
    max_seq_length: int = 4096
)
```

### Core Methods

#### generate()
Generate text for given prompts.

```python
results = llm.generate(
    prompts=["Hello, world!", "How are you?"],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=50)
)
```

#### generate_stream()
Generate text with streaming output.

```python
for result in llm.generate_stream("Tell me a story"):
    print(result["token"], end="", flush=True)
    if result["finished"]:
        break
```

#### chat()
Generate chat responses using structured messages.

```python
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
]
result = llm.chat(messages)
```

### Convenience Methods

```python
# Greedy decoding
results = llm.generate_greedy("Hello", max_tokens=50)

# Creative generation  
results = llm.generate_creative("Write a poem", max_tokens=100)

# Balanced generation
results = llm.generate_balanced("Explain AI", max_tokens=200)
```

## SamplingParams Class

Configuration for text generation behavior.

### Constructor

```python
from nano_qwen3_serving import SamplingParams

params = SamplingParams(
    temperature=1.0,              # Randomness (0.0-2.0)
    top_p=1.0,                   # Nucleus sampling (0.0-1.0)  
    top_k=-1,                    # Top-k sampling (-1 for no limit)
    max_tokens=100,              # Maximum tokens to generate
    min_tokens=0,                # Minimum tokens to generate
    stop_sequences=None,         # Stop sequences
    repetition_penalty=1.0,      # Repetition penalty
    do_sample=True,              # Use sampling vs greedy
    use_beam_search=False,       # Enable beam search
    num_beams=1                  # Number of beams
)
```

### Class Methods

```python
# Preset configurations
SamplingParams.greedy(max_tokens=100)      # Deterministic
SamplingParams.creative(max_tokens=100)    # High creativity
SamplingParams.balanced(max_tokens=100)    # Balanced approach
```
