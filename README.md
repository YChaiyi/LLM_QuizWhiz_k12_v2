# QuizWhiz_k12_v2

基于 Qwen3-32B 微调的 K12 教育问答模型 - 多轮对话工具

[![ModelScope](https://img.shields.io/badge/ModelScope-YChaiyi%2FQuizWhiz__k12__v2-blue)](https://modelscope.cn/models/YChaiyi/QuizWhiz_k12_v2)
[![GitHub](https://img.shields.io/badge/GitHub-YChaiyi%2FLLM__QuizWhiz__k12__v2-green)](https://github.com/YChaiyi/LLM_QuizWhiz_k12_v2)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

## 简介

QuizWhiz_k12_v2 是一个专门用于 K12 教育场景的智能问答模型，通过 LoRA 微调 Qwen3-32B 基础模型训练而成。本项目提供即开即用的多轮对话功能，**支持从 ModelScope 自动下载模型**。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行对话（自动下载模型）

**最简单的使用方式**，首次运行会自动从 ModelScope 下载模型：

```bash
python inference.py
```

模型将自动下载到 `./models/` 目录，后续运行会直接使用缓存。

### 3. 其他启动方式

**使用 8bit 量化**（推荐，节省显存）:
```bash
python inference.py --load_in_8bit
```

**使用 4bit 量化**（显存严重不足时）:
```bash
python inference.py --load_in_4bit
```

**使用本地模型**（如果已有模型文件）:
```bash
python inference.py --model_path /path/to/your/model
```

**使用快捷脚本**:
```bash
bash scripts/run_chat.sh --load_in_8bit
```

## 对话示例

```
============================================================
QuizWhiz_k12_v2 多轮对话系统
============================================================

使用说明：
  - 直接输入问题开始对话
  - 输入 'clear' 清空对话历史
  - 输入 'exit' 或 'quit' 退出

============================================================

用户: 什么是光合作用？

QuizWhiz: 光合作用是植物、藻类和某些细菌利用光能，将二氧化碳和水转化为
有机物（如葡萄糖），并释放氧气的过程。这是地球上最重要的生化反应之一...

用户: 它需要哪些条件？

QuizWhiz: 光合作用需要以下基本条件：
1. 光照：提供能量
2. 二氧化碳：作为碳源
3. 水：提供氢和氧
4. 叶绿素：吸收光能
5. 适宜的温度：通常在20-30°C之间效果最好...
```

## Python API 使用

### 基本用法（自动下载）

```python
from inference import QuizWhizChat

# 初始化（首次运行会自动下载模型）
chat_bot = QuizWhizChat(load_in_8bit=True)

# 多轮对话
messages = []

# 第一轮
messages.append({"role": "user", "content": "什么是DNA？"})
response = chat_bot.chat(messages)
print(response)
messages.append({"role": "assistant", "content": response})

# 第二轮
messages.append({"role": "user", "content": "它的结构是怎样的？"})
response = chat_bot.chat(messages)
print(response)
```

### 使用本地模型

```python
from inference import QuizWhizChat

# 使用本地模型路径
chat_bot = QuizWhizChat(
    model_path="./models/QuizWhiz_k12_v2",
    load_in_8bit=True
)

messages = [{"role": "user", "content": "解释牛顿第一定律"}]
response = chat_bot.chat(messages)
print(response)
```

### 禁用自动下载

```python
from inference import QuizWhizChat

# 如果不想自动下载，必须提供本地路径
chat_bot = QuizWhizChat(
    model_path="/path/to/local/model",
    auto_download=False
)
```

## 环境要求

### 硬件要求

- **GPU 推荐配置**:
  - 完整精度 (FP16/BF16): ≥64GB 显存
  - 8bit 量化: ≥32GB 显存（推荐）
  - 4bit 量化: ≥16GB 显存
- **CPU**: 如使用 CPU 推理，建议 ≥32GB 内存
- **磁盘空间**: 首次下载模型需要约 65GB 空间

### 软件要求

- Python ≥ 3.8
- CUDA ≥ 11.8 (GPU 推理)
- PyTorch ≥ 2.0.0

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | str | `YChaiyi/QuizWhiz_k12_v2` | 模型路径或 ModelScope ID |
| `--device` | str | `auto` | 运行设备 (auto/cuda/cpu) |
| `--load_in_8bit` | flag | False | 使用 8bit 量化 |
| `--load_in_4bit` | flag | False | 使用 4bit 量化 |
| `--no_auto_download` | flag | False | 禁用自动下载 |

## 使用说明

### 交互命令

- 直接输入问题进行对话
- 输入 `clear` 清空对话历史
- 输入 `exit` 或 `quit` 退出程序

### 模型下载

- **自动下载**: 默认启用，首次运行自动从 ModelScope 下载
- **下载位置**: `./models/` 目录
- **缓存机制**: 下载后会缓存，后续运行直接使用本地文件
- **手动下载**: 也可以手动从 [ModelScope](https://modelscope.cn/models/YChaiyi/QuizWhiz_k12_v2) 下载

### 量化选择

- **不使用量化**: 精度最高，显存需求最大（~64GB）
- **8bit 量化**: 显存减半（~32GB），精度略有下降，**推荐使用**
- **4bit 量化**: 显存最小（~16GB），精度有一定损失

## 模型信息

- **模型名称**: QuizWhiz_k12_v2
- **基础模型**: Qwen/Qwen3-32B
- **微调方法**: LoRA
- **训练场景**: K12 教育（数学、物理、化学、生物等）
- **支持语言**: 中文
- **参数量**: 32B
- **ModelScope**: [YChaiyi/QuizWhiz_k12_v2](https://modelscope.cn/models/YChaiyi/QuizWhiz_k12_v2)
- **GitHub**: [YChaiyi/LLM_QuizWhiz_k12_v2](https://github.com/YChaiyi/LLM_QuizWhiz_k12_v2)

## 许可证

本项目继承 Qwen3-32B 基础模型的 Apache-2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 常见问题

**Q: 首次运行需要多长时间？**
A: 首次运行需要下载约 65GB 的模型文件，取决于网络速度，通常需要 10-30 分钟。后续运行会直接使用缓存。

**Q: 如何查看下载进度？**
A: ModelScope 会在终端显示下载进度。如果下载中断，重新运行脚本会继续下载。

**Q: 显存不足怎么办？**
A: 使用 `--load_in_8bit` 或 `--load_in_4bit` 参数启用量化。

**Q: 如何手动下载模型？**
A: 访问 [ModelScope 模型页面](https://modelscope.cn/models/YChaiyi/QuizWhiz_k12_v2)，或使用：
```python
from modelscope import snapshot_download
snapshot_download('YChaiyi/QuizWhiz_k12_v2', cache_dir='./models')
```

**Q: 可以在 CPU 上运行吗？**
A: 可以，使用 `--device cpu` 参数，但推理速度会很慢，且需要大内存（≥32GB）。

**Q: 模型支持哪些学科？**
A: 主要支持 K12 阶段的数学、物理、化学、生物等基础学科问答。

## 引用

如果您在研究中使用了本模型，请引用：

```bibtex
@misc{quizwhiz_k12_v2,
  title={QuizWhiz_k12_v2: A Fine-tuned LLM for K12 Education},
  author={YChaiyi},
  year={2024},
  publisher={ModelScope},
  howpublished={\url{https://modelscope.cn/models/YChaiyi/QuizWhiz_k12_v2}}
}
```

## 致谢

- 基础模型: [Qwen/Qwen3-32B](https://github.com/QwenLM/Qwen)
- 模型托管: [ModelScope](https://modelscope.cn/)
