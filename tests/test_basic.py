#!/usr/bin/env python3
"""
Basic tests for nano-qwen3-serving.
"""

import pytest
import requests
import time


def test_health_endpoint():
    """Test health endpoint."""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_info" in data
        assert "uptime" in data
    except requests.exceptions.ConnectionError:
        pytest.skip("Service not running")


def test_models_endpoint():
    """Test models endpoint."""
    try:
        response = requests.get("http://127.0.0.1:8000/v1/models", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) > 0
    except requests.exceptions.ConnectionError:
        pytest.skip("Service not running")


def test_chat_completion():
    """Test chat completion endpoint."""
    try:
        response = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={
                "model": "qwen3-0.6b",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "max_tokens": 10
            },
            timeout=30
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
    except requests.exceptions.ConnectionError:
        pytest.skip("Service not running")


def test_streaming_chat_completion():
    """Test streaming chat completion."""
    try:
        response = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={
                "model": "qwen3-0.6b",
                "messages": [
                    {"role": "user", "content": "Hi"}
                ],
                "stream": True,
                "max_tokens": 10
            },
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=30
        )
        assert response.status_code == 200
        
        # Check that we get streaming data
        chunks = []
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    chunks.append(data_str)
        
        assert len(chunks) > 0
    except requests.exceptions.ConnectionError:
        pytest.skip("Service not running")


def test_stats_endpoint():
    """Test stats endpoint."""
    try:
        response = requests.get("http://127.0.0.1:8000/stats", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "uptime" in data
        assert "total_requests" in data
        assert "completed_requests" in data
    except requests.exceptions.ConnectionError:
        pytest.skip("Service not running")


if __name__ == "__main__":
    # Run tests manually
    print("Running basic tests...")
    
    tests = [
        test_health_endpoint,
        test_models_endpoint,
        test_chat_completion,
        test_streaming_chat_completion,
        test_stats_endpoint,
    ]
    
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__} passed")
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
    
    print("Tests completed!") 