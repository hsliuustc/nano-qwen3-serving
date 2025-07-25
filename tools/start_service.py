#!/usr/bin/env python3
"""
Service launcher for the nano Qwen3 serving engine.
"""

import argparse
import uvicorn
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nano_qwen3_serving.service.openai_server import app


def main():
    """Main function to start the service."""
    parser = argparse.ArgumentParser(
        description="Start the nano Qwen3 serving service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Start with default settings
  %(prog)s --host 0.0.0.0     # Bind to all interfaces
  %(prog)s --port 8080        # Use custom port
  %(prog)s --reload           # Enable auto-reload for development
  %(prog)s --workers 4        # Use multiple workers
        """
    )
    
    # Server configuration
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Model name or path (default: Qwen/Qwen3-0.6B)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use"
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Data type (default: float16)"
    )
    
    # Engine configuration
    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=1000,
        help="Maximum queue size (default: 1000)"
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=1024,
        help="Number of memory blocks (default: 1024)"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size (default: 16)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)"
    )
    
    args = parser.parse_args()
    
    # Print startup information
    print("🚀 Starting nano Qwen3 Serving Service")
    print(f"📊 Model: {args.model}")
    print(f"🔧 Device: {args.device}")
    print(f"🌐 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"👥 Workers: {args.workers}")
    print(f"📝 Log Level: {args.log_level}")
    print("-" * 50)
    
    # Configure uvicorn
    uvicorn_config = {
        "app": "nano_qwen3_serving.service.openai_server:app",
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
        "workers": args.workers if not args.reload else 1,
        "log_level": args.log_level,
        "access_log": True,
    }
    
    # Start the server
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        print("\n👋 Service stopped by user")
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 