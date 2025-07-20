#!/usr/bin/env python3
"""
Basic usage example for nano-qwen3-serving.
"""

import requests
import json


def basic_chat_example():
    """Basic chat completion example."""
    print("🚀 Basic Chat Completion Example")
    print("=" * 50)
    
    # Make a simple chat request
    response = requests.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        json={
            "model": "qwen3-0.6b",
            "messages": [
                {"role": "user", "content": "Hello! Can you tell me a joke?"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Tokens used: {result['usage']['total_tokens']}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def streaming_example():
    """Streaming chat completion example."""
    print("\n🌊 Streaming Chat Completion Example")
    print("=" * 50)
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={
                "model": "qwen3-0.6b",
                "messages": [
                    {"role": "user", "content": "Write a short story about a robot learning to paint."}
                ],
                "stream": True,
                "max_tokens": 100,
                "temperature": 0.8
            },
            headers={"Accept": "text/event-stream"},
            stream=True
        )
        
        print("Streaming response:")
        full_response = ""
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        if 'choices' in chunk and chunk['choices']:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                print(content, end='', flush=True)
                                full_response += content
                    except json.JSONDecodeError:
                        continue
        
        print(f"\n\nFull response: {full_response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def health_check_example():
    """Health check example."""
    print("\n🏥 Health Check Example")
    print("=" * 50)
    
    try:
        response = requests.get("http://127.0.0.1:8000/health")
        if response.status_code == 200:
            health = response.json()
            print("✅ Service is healthy!")
            print(f"Status: {health['status']}")
            print(f"Uptime: {health['uptime']:.2f} seconds")
            print(f"Model: {health['model_info']['model_name']}")
        else:
            print(f"❌ Service health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")


def list_models_example():
    """List models example."""
    print("\n📋 List Models Example")
    print("=" * 50)
    
    try:
        response = requests.get("http://127.0.0.1:8000/v1/models")
        if response.status_code == 200:
            models = response.json()
            print("Available models:")
            for model in models['data']:
                print(f"  - {model['id']} (owned by {model['owned_by']})")
        else:
            print(f"❌ Failed to list models: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all examples."""
    print("🎯 Nano Qwen3 Serving - Basic Usage Examples")
    print("=" * 60)
    print("Make sure the service is running on http://127.0.0.1:8000")
    print("Start it with: python tools/start_service.py")
    print("=" * 60)
    
    # Run examples
    health_check_example()
    list_models_example()
    basic_chat_example()
    streaming_example()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("\n💡 Tips:")
    print("- The API is OpenAI-compatible")
    print("- Use streaming for real-time responses")
    print("- Check /health for service status")
    print("- Use /stats for performance monitoring")


if __name__ == "__main__":
    main() 