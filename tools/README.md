# Development Tools

This directory contains development and testing tools for the nano Qwen3 serving engine.

## Available Tools

### 🚀 Performance Testing

#### `performance_test.py`
Comprehensive performance testing script that evaluates the system across different token limits and prompts.

**Usage:**
```bash
python tools/performance_test.py
```

**Features:**
- Tests multiple token limits (10, 20, 50, 100 tokens)
- Multiple test prompts covering different topics
- Detailed performance metrics collection
- Results saved to `results/performance_results.json`
- Comprehensive system statistics

**Output:**
- Real-time performance metrics
- Average speed calculations
- Memory usage statistics
- Model inference timing
- Detailed JSON results file

### 💻 Command Line Interface

#### `cli.py`
Interactive command-line interface for testing the nano Qwen3 serving engine.

**Usage:**
```bash
# Interactive mode
python tools/cli.py -i

# Single prompt
python tools/cli.py "What is AI?"

# Fast mode (50 tokens max)
python tools/cli.py --fast "Explain quantum computing"

# Custom parameters
python tools/cli.py --max-tokens 30 --temperature 0.8 "Write a haiku"
```

**Features:**
- Interactive chat mode
- Single prompt processing
- Batch processing from files
- Multiple sampling strategies (greedy, creative, balanced)
- Performance statistics
- Model information display

**Commands in Interactive Mode:**
- `help` - Show available commands
- `stats` - Display system statistics
- `clear` - Clear statistics
- `quit` - Exit the program

## Tool Configuration

### Performance Test Configuration
The performance test can be customized by modifying the test parameters in `performance_test.py`:

```python
# Test prompts
test_prompts = [
    "What is AI?",
    "Explain quantum computing",
    # Add your own prompts...
]

# Token limits to test
token_limits = [10, 20, 50, 100]
```

### CLI Configuration
The CLI supports various command-line arguments:

```bash
# Model configuration
--model Qwen/Qwen3-0.6B    # Model name
--device mps               # Device (mps, cpu, cuda)
--dtype float16            # Data type

# Sampling parameters
--temperature 0.7          # Sampling temperature
--max-tokens 100           # Maximum tokens to generate
--top-p 0.9               # Top-p sampling
--top-k 50                # Top-k sampling

# Sampling presets
--greedy                  # Greedy decoding
--creative                # Creative sampling
--balanced                # Balanced sampling
--fast                    # Fast mode (50 tokens, greedy)
```

## Results

Performance test results are automatically saved to:
- `results/performance_results.json` - Detailed performance data
- Console output - Real-time metrics and summary

## Development Workflow

1. **Testing Changes**: Use `performance_test.py` to validate performance impact
2. **Interactive Testing**: Use `cli.py` for manual testing and exploration
3. **Results Analysis**: Review `results/performance_results.json` for detailed metrics
4. **Iteration**: Modify code and re-run tests to validate improvements

## Troubleshooting

### Common Issues

**Performance Test Fails:**
- Ensure the model is properly loaded
- Check MPS availability on Apple Silicon
- Verify all dependencies are installed

**CLI Issues:**
- Check model path and availability
- Verify device configuration
- Ensure proper conda environment activation

### Debug Mode

Enable verbose logging for debugging:
```bash
python tools/cli.py -v -i
python tools/performance_test.py  # Debug logging included
```

## Contributing

When adding new tools:
1. Follow the existing naming conventions
2. Include comprehensive documentation
3. Add appropriate error handling
4. Include usage examples
5. Update this README with new tool information 