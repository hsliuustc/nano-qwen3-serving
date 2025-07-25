#!/usr/bin/env python3

from nano_qwen3_serving import LLM, SamplingParams


# 初始化模型
llm = LLM(
    model_name="/zx_data1/nano-vllm/models/Qwen3-0.6B", 
    device="auto",
    max_seq_length=256  # 减少序列长度
)

# 执行推理
result = llm.generate_single("tell me about the history of the world", SamplingParams(max_tokens=10))

# 输出结果
print(f"输入: tell me about the history of the world")
print(f"输出: {result['generated_text']}")

# 清理
llm.shutdown()