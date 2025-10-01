#!/usr/bin/env python3
"""
QuizWhiz_k12_v2 多轮对话推理脚本
基于Qwen3-32B微调的K12教育问答模型
"""

import os
import torch
import argparse
from typing import List, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)


def download_model_from_modelscope(model_id: str, cache_dir: str = "./models") -> str:
    """
    从ModelScope下载模型

    Args:
        model_id: ModelScope模型ID
        cache_dir: 本地缓存目录

    Returns:
        模型本地路径
    """
    try:
        from modelscope import snapshot_download

        print(f"正在从ModelScope下载模型: {model_id}")
        print(f"缓存目录: {cache_dir}")
        print("首次下载可能需要较长时间，请耐心等待...\n")

        model_dir = snapshot_download(
            model_id,
            cache_dir=cache_dir,
            revision='master'
        )

        print(f"✓ 模型下载完成: {model_dir}\n")
        return model_dir

    except ImportError:
        raise ImportError(
            "未安装modelscope库，请运行: pip install modelscope"
        )
    except Exception as e:
        raise RuntimeError(f"模型下载失败: {e}")


class QuizWhizChat:
    """QuizWhiz_k12_v2 多轮对话类"""

    def __init__(
        self,
        model_path: str = None,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        auto_download: bool = True,
    ):
        """
        初始化对话模型

        Args:
            model_path: 模型路径（本地路径或ModelScope ID）
            device: 运行设备 (auto/cuda/cpu)
            load_in_8bit: 是否使用8bit量化
            load_in_4bit: 是否使用4bit量化
            auto_download: 是否自动从ModelScope下载
        """
        # 默认使用ModelScope模型ID
        if model_path is None:
            model_path = "YChaiyi/QuizWhiz_k12_v2"

        # 检查是否为本地路径
        if os.path.exists(model_path):
            self.model_path = model_path
            print(f"使用本地模型: {model_path}")
        elif auto_download:
            # 从ModelScope下载
            self.model_path = download_model_from_modelscope(model_path)
        else:
            raise FileNotFoundError(
                f"模型路径不存在: {model_path}\n"
                "提示: 使用 --auto_download 参数从ModelScope自动下载"
            )

        print(f"正在加载模型...")
        print(f"设备: {device}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # 加载模型
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": "auto",
            "device_map": device if device != "auto" else "auto",
        }

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            print("使用8bit量化")
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            print("使用4bit量化")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs
        )

        # 生成配置
        self.generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            max_new_tokens=2048,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        print("模型加载完成！\n")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        多轮对话生成

        Args:
            messages: 对话历史 [{"role": "user/assistant", "content": "..."}]

        Returns:
            生成的回复
        """
        # 格式化对话
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            text += "<|im_start|>assistant\n"

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(self.model.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )

        # 解码
        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        return response.strip()

    def interactive_chat(self):
        """交互式多轮对话"""
        print("=" * 60)
        print("QuizWhiz_k12_v2 多轮对话系统")
        print("=" * 60)
        print("\n使用说明：")
        print("  - 直接输入问题开始对话")
        print("  - 输入 'clear' 清空对话历史")
        print("  - 输入 'exit' 或 'quit' 退出\n")
        print("=" * 60 + "\n")

        messages = []

        try:
            while True:
                user_input = input("用户: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    print("\n再见！")
                    break

                if user_input.lower() == 'clear':
                    messages = []
                    print("\n对话历史已清空\n")
                    continue

                # 添加用户消息
                messages.append({"role": "user", "content": user_input})

                # 生成回复
                print("\nQuizWhiz: ", end="", flush=True)
                response = self.chat(messages)
                print(response + "\n")

                # 添加助手回复
                messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\n对话已中断")


def main():
    parser = argparse.ArgumentParser(
        description="QuizWhiz_k12_v2 多轮对话工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动从ModelScope下载并运行
  python inference.py

  # 使用本地模型
  python inference.py --model_path ./models/QuizWhiz_k12_v2

  # 使用8bit量化
  python inference.py --load_in_8bit

  # 禁用自动下载
  python inference.py --no_auto_download --model_path /path/to/model
        """
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型路径（本地路径或ModelScope模型ID，默认: YChaiyi/QuizWhiz_k12_v2）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="运行设备"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="使用8bit量化"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="使用4bit量化"
    )
    parser.add_argument(
        "--no_auto_download",
        action="store_true",
        help="禁用从ModelScope自动下载"
    )

    args = parser.parse_args()

    # 初始化对话系统
    try:
        chat_bot = QuizWhizChat(
            model_path=args.model_path,
            device=args.device,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            auto_download=not args.no_auto_download
        )

        # 启动交互式对话
        chat_bot.interactive_chat()

    except Exception as e:
        print(f"\n错误: {e}")
        print("\n提示:")
        print("  1. 确保已安装依赖: pip install -r requirements.txt")
        print("  2. 如需从ModelScope下载，请确保网络连接正常")
        print("  3. 如使用本地模型，请检查模型路径是否正确")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
