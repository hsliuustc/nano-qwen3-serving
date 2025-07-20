#!/bin/bash

# Deploy Documentation Script
# This script builds and serves the documentation locally

set -e

echo "🚀 Building Nano Qwen3 Serving Documentation..."

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo "❌ MkDocs is not installed. Installing..."
    pip install mkdocs mkdocs-material "mkdocstrings[python]" mkdocs-autorefs
fi

# Build documentation
echo "📚 Building documentation..."
mkdocs build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Documentation built successfully!"
    echo "📁 Documentation is available in the 'site' directory"
    
    # Ask if user wants to serve the documentation
    read -p "🌐 Would you like to serve the documentation locally? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Starting documentation server at http://localhost:8000"
        echo "Press Ctrl+C to stop the server"
        mkdocs serve
    fi
else
    echo "❌ Documentation build failed!"
    exit 1
fi 