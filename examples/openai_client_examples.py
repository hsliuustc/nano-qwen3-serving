#!/usr/bin/env python3
"""
Comprehensive examples for using the OpenAI-compatible nano Qwen3 serving API.
"""

import asyncio
import json
import time
import requests
import aiohttp
import websockets
from typing import Dict, Any, List


class OpenAICompatibleClient:
    """Client for the OpenAI-compatible nano Qwen3 serving API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url.rstrip('/')
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send a chat completion request."""
        url = f"{self.base_url}/v1/chat/completions"
        data = {
            "model": "qwen3-0.6b",
            "messages": messages,
            **kwargs
        }
        
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        return response.json()
    
    def chat_completion_stream(self, messages: List[Dict[str, str]], **kwargs):
        """Send a streaming chat completion request."""
        url = f"{self.base_url}/v1/chat/completions"
        data = {
            "model": "qwen3-0.6b",
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        response = requests.post(url, json=data, headers=headers, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    if data_str == '[DONE]':
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
    
    def completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Send a completion request (legacy API)."""
        url = f"{self.base_url}/v1/completions"
        data = {
            "model": "qwen3-0.6b",
            "prompt": prompt,
            **kwargs
        }
        
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        url = f"{self.base_url}/v1/models"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        url = f"{self.base_url}/stats"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


class AsyncOpenAICompatibleClient:
    """Async client for the OpenAI-compatible nano Qwen3 serving API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url.rstrip('/')
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send an async chat completion request."""
        url = f"{self.base_url}/v1/chat/completions"
        data = {
            "model": "qwen3-0.6b",
            "messages": messages,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                response.raise_for_status()
                return await response.json()
    
    async def chat_completion_stream(self, messages: List[Dict[str, str]], **kwargs):
        """Send an async streaming chat completion request."""
        url = f"{self.base_url}/v1/chat/completions"
        data = {
            "model": "qwen3-0.6b",
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        headers = {"Accept": "text/event-stream"}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            break
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models."""
        url = f"{self.base_url}/v1/models"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()


def example_1_basic_chat():
    """Example 1: Basic chat completion."""
    print("=" * 60)
    print("Example 1: Basic Chat Completion")
    print("=" * 60)
    
    client = OpenAICompatibleClient()
    
    messages = [
        {"role": "user", "content": "Hello! Can you tell me a short joke?"}
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        
        print("Response:")
        print(json.dumps(response, indent=2))
        
        # Extract the generated text
        generated_text = response['choices'][0]['message']['content']
        print(f"\nGenerated text: {generated_text}")
        
        # Show token usage
        usage = response['usage']
        print(f"Token usage: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total")
        
    except Exception as e:
        print(f"Error: {e}")


def example_2_streaming_chat():
    """Example 2: Streaming chat completion."""
    print("\n" + "=" * 60)
    print("Example 2: Streaming Chat Completion")
    print("=" * 60)
    
    client = OpenAICompatibleClient()
    
    messages = [
        {"role": "user", "content": "Write a short story about a robot learning to paint."}
    ]
    
    try:
        print("Streaming response:")
        full_response = ""
        
        for chunk in client.chat_completion_stream(
            messages=messages,
            max_tokens=150,
            temperature=0.8
        ):
            if 'choices' in chunk and chunk['choices']:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    content = delta['content']
                    print(content, end='', flush=True)
                    full_response += content
        
        print(f"\n\nFull response: {full_response}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_3_conversation():
    """Example 3: Multi-turn conversation."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-turn Conversation")
    print("=" * 60)
    
    client = OpenAICompatibleClient()
    
    conversation = [
        {"role": "system", "content": "You are a helpful assistant who likes to explain things clearly."},
        {"role": "user", "content": "What is machine learning?"},
    ]
    
    try:
        # First turn
        response = client.chat_completion(
            messages=conversation,
            max_tokens=100,
            temperature=0.7
        )
        
        assistant_response = response['choices'][0]['message']['content']
        print(f"User: {conversation[-1]['content']}")
        print(f"Assistant: {assistant_response}")
        
        # Add assistant's response to conversation
        conversation.append({"role": "assistant", "content": assistant_response})
        
        # Second turn
        conversation.append({"role": "user", "content": "Can you give me a simple example?"})
        
        response = client.chat_completion(
            messages=conversation,
            max_tokens=100,
            temperature=0.7
        )
        
        assistant_response = response['choices'][0]['message']['content']
        print(f"User: {conversation[-1]['content']}")
        print(f"Assistant: {assistant_response}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_4_legacy_completion():
    """Example 4: Legacy completion API."""
    print("\n" + "=" * 60)
    print("Example 4: Legacy Completion API")
    print("=" * 60)
    
    client = OpenAICompatibleClient()
    
    prompt = "The future of artificial intelligence is"
    
    try:
        response = client.completion(
            prompt=prompt,
            max_tokens=50,
            temperature=0.8
        )
        
        print("Response:")
        print(json.dumps(response, indent=2))
        
        generated_text = response['choices'][0]['text']
        print(f"\nGenerated text: {generated_text}")
        
    except Exception as e:
        print(f"Error: {e}")


async def example_5_async_client():
    """Example 5: Async client usage."""
    print("\n" + "=" * 60)
    print("Example 5: Async Client Usage")
    print("=" * 60)
    
    client = AsyncOpenAICompatibleClient()
    
    messages = [
        {"role": "user", "content": "What are the benefits of async programming?"}
    ]
    
    try:
        response = await client.chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        
        print("Async response:")
        print(json.dumps(response, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")


def example_6_service_info():
    """Example 6: Service information."""
    print("\n" + "=" * 60)
    print("Example 6: Service Information")
    print("=" * 60)
    
    client = OpenAICompatibleClient()
    
    try:
        # List models
        models = client.list_models()
        print("Available models:")
        print(json.dumps(models, indent=2))
        
        # Health check
        health = client.health_check()
        print("\nHealth status:")
        print(json.dumps(health, indent=2))
        
        # Service stats
        stats = client.get_stats()
        print("\nService statistics:")
        print(json.dumps(stats, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")


def example_7_curl_commands():
    """Example 7: Equivalent curl commands."""
    print("\n" + "=" * 60)
    print("Example 7: Equivalent Curl Commands")
    print("=" * 60)
    
    print("1. Basic Chat Completion:")
    print("""curl -X POST http://127.0.0.1:8001/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'""")
    
    print("\n2. Streaming Chat Completion:")
    print("""curl -X POST http://127.0.0.1:8001/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Accept: text/event-stream" \\
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true,
    "max_tokens": 100
  }'""")
    
    print("\n3. List Models:")
    print("curl -X GET http://127.0.0.1:8001/v1/models")
    
    print("\n4. Health Check:")
    print("curl -X GET http://127.0.0.1:8001/health")
    
    print("\n5. Legacy Completion:")
    print("""curl -X POST http://127.0.0.1:8001/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen3-0.6b",
    "prompt": "The quick brown fox",
    "max_tokens": 30,
    "temperature": 0.7
  }'""")


def main():
    """Run all examples."""
    print("🚀 OpenAI-Compatible Nano Qwen3 Serving API Examples")
    print("=" * 80)
    
    # Run synchronous examples
    example_1_basic_chat()
    example_2_streaming_chat()
    example_3_conversation()
    example_4_legacy_completion()
    example_6_service_info()
    example_7_curl_commands()
    
    # Run async example
    asyncio.run(example_5_async_client())
    
    print("\n" + "=" * 80)
    print("✅ All examples completed!")
    print("\n💡 Tips:")
    print("- Use streaming for real-time token generation")
    print("- The API is OpenAI-compatible, so you can use existing OpenAI clients")
    print("- Check /health for service status")
    print("- Use /stats for performance monitoring")


if __name__ == "__main__":
    main() 