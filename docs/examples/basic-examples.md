# Basic Examples

This guide provides practical examples of how to use Nano Qwen3 Serving for common use cases.

## 🚀 Getting Started

### Prerequisites

1. **Start the server:**
   ```bash
   python -m nano_qwen3_serving --port 8000
   ```

2. **Install required packages:**
   ```bash
   pip install requests openai
   ```

## 📝 Basic Chat Completion

### Simple Request

```python
import requests

def simple_chat():
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "messages": [
                {"role": "user", "content": "Hello! How are you today?"}
            ],
            "max_tokens": 100
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(result["choices"][0]["message"]["content"])
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Run the example
simple_chat()
```

### Multi-turn Conversation

```python
import requests

def multi_turn_conversation():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is the population of Paris?"}
    ]
    
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(result["choices"][0]["message"]["content"])
    else:
        print(f"Error: {response.status_code}")

multi_turn_conversation()
```

## 🔄 Streaming Responses

### Basic Streaming

```python
import requests
import json

def streaming_chat():
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "messages": [
                {"role": "user", "content": "Write a short story about a robot."}
            ],
            "stream": True,
            "max_tokens": 200
        },
        stream=True
    )
    
    print("Response: ", end="", flush=True)
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                    if 'choices' in chunk and chunk['choices']:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            print(delta['content'], end='', flush=True)
                except json.JSONDecodeError:
                    continue
    
    print()  # New line at the end

streaming_chat()
```

### Streaming with Progress Tracking

```python
import requests
import json
import time

def streaming_with_progress():
    start_time = time.time()
    
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "messages": [
                {"role": "user", "content": "Explain quantum computing in detail."}
            ],
            "stream": True,
            "max_tokens": 300
        },
        stream=True
    )
    
    tokens_received = 0
    print("Response: ", end="", flush=True)
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                    if 'choices' in chunk and chunk['choices']:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            content = delta['content']
                            print(content, end='', flush=True)
                            tokens_received += 1
                            
                            # Show progress every 10 tokens
                            if tokens_received % 10 == 0:
                                elapsed = time.time() - start_time
                                rate = tokens_received / elapsed
                                print(f"\n[Progress: {tokens_received} tokens, {rate:.1f} tokens/s]")
                except json.JSONDecodeError:
                    continue
    
    elapsed = time.time() - start_time
    print(f"\n\nCompleted in {elapsed:.2f}s with {tokens_received} tokens")

streaming_with_progress()
```

## 🎛️ Advanced Parameters

### Temperature and Sampling

```python
import requests

def temperature_examples():
    prompts = [
        "Write a creative story about a cat.",
        "Write a factual explanation of photosynthesis.",
        "Write a poem about the ocean."
    ]
    
    temperatures = [0.1, 0.7, 1.2]
    
    for prompt, temp in zip(prompts, temperatures):
        print(f"\n--- Temperature: {temp} ---")
        print(f"Prompt: {prompt}")
        
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "Qwen/Qwen3-0.6B",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": temp,
                "top_p": 0.9
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['choices'][0]['message']['content']}")
        else:
            print(f"Error: {response.status_code}")

temperature_examples()
```

### Stop Sequences

```python
import requests

def stop_sequences_example():
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "messages": [
                {"role": "user", "content": "List the first 5 planets in our solar system:"}
            ],
            "max_tokens": 200,
            "stop": ["\n\n", "6.", "7.", "8.", "9."]
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(result["choices"][0]["message"]["content"])
    else:
        print(f"Error: {response.status_code}")

stop_sequences_example()
```

## 🔧 Using OpenAI Client

### OpenAI-Compatible Client

```python
import openai

# Configure the client to use your local server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not used for local server
)

def openai_client_example():
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-0.6B",
            messages=[
                {"role": "user", "content": "What is the meaning of life?"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error: {e}")

openai_client_example()
```

### Streaming with OpenAI Client

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

def openai_streaming_example():
    try:
        stream = client.chat.completions.create(
            model="Qwen/Qwen3-0.6B",
            messages=[
                {"role": "user", "content": "Write a haiku about programming."}
            ],
            stream=True,
            max_tokens=100
        )
        
        print("Response: ", end="", flush=True)
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        
        print()  # New line at the end
        
    except Exception as e:
        print(f"Error: {e}")

openai_streaming_example()
```

## 📊 Error Handling

### Robust Error Handling

```python
import requests
import time

def robust_chat_request(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/v1/chat/completions",
                json={
                    "model": "Qwen/Qwen3-0.6B",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100
                },
                timeout=30  # 30 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"HTTP Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
        except requests.exceptions.ConnectionError:
            print(f"Connection error on attempt {attempt + 1}")
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
        
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    return "Failed to get response after all retries"

# Test the robust function
result = robust_chat_request("What is artificial intelligence?")
print(f"Result: {result}")
```

## 🔍 Health and Status

### Check Server Health

```python
import requests

def check_server_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print("Server Status:", health_data["status"])
            print("Model:", health_data["model"])
            print("Device:", health_data["device"])
            print("Uptime:", health_data["uptime"], "seconds")
            return True
        else:
            print(f"Server unhealthy: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Server unreachable: {e}")
        return False

check_server_health()
```

### Get Performance Stats

```python
import requests

def get_performance_stats():
    try:
        response = requests.get("http://localhost:8000/stats", timeout=5)
        
        if response.status_code == 200:
            stats = response.json()
            print("Performance Statistics:")
            print(f"  Requests processed: {stats['requests_processed']}")
            print(f"  Tokens generated: {stats['tokens_generated']}")
            print(f"  Average response time: {stats['average_response_time']:.3f}s")
            print(f"  Requests per second: {stats['requests_per_second']:.1f}")
            print(f"  Memory usage: {stats['memory_usage_mb']} MB")
            print(f"  GPU utilization: {stats['gpu_utilization']:.1%}")
        else:
            print(f"Failed to get stats: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error getting stats: {e}")

get_performance_stats()
```

## 🎯 Complete Example Application

### Simple Chat Application

```python
import requests
import json
import sys

class NanoQwen3Chat:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.messages = []
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def chat(self, user_input, stream=False):
        self.add_message("user", user_input)
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "Qwen/Qwen3-0.6B",
                    "messages": self.messages,
                    "stream": stream,
                    "max_tokens": 200,
                    "temperature": 0.7
                },
                stream=stream,
                timeout=30
            )
            
            if response.status_code == 200:
                if stream:
                    return self._handle_streaming_response(response)
                else:
                    result = response.json()
                    assistant_message = result["choices"][0]["message"]["content"]
                    self.add_message("assistant", assistant_message)
                    return assistant_message
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error: {e}"
    
    def _handle_streaming_response(self, response):
        full_response = ""
        print("Assistant: ", end="", flush=True)
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and chunk['choices']:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                print(content, end='', flush=True)
                                full_response += content
                    except json.JSONDecodeError:
                        continue
        
        print()  # New line
        self.add_message("assistant", full_response)
        return full_response
    
    def interactive_chat(self):
        print("Nano Qwen3 Chat (type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = self.chat(user_input, stream=True)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break

# Run the interactive chat
if __name__ == "__main__":
    chat = NanoQwen3Chat()
    chat.interactive_chat()
```

## 📚 Next Steps

- **[Advanced Examples](advanced-examples.md)**: More complex use cases
- **[Client Libraries](client-libraries.md)**: Integration with different frameworks
- **[API Reference](../user-guide/api-reference.md)**: Complete API documentation
- **[Troubleshooting](../troubleshooting/common-issues.md)**: Common issues and solutions

---

**💡 Tip:** These examples can be run directly or modified for your specific use case. Make sure the server is running before executing any examples! 