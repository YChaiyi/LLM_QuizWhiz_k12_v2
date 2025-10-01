#!/bin/bash

# QuizWhiz_k12_v2 快速启动脚本

MODEL_PATH="./models/QuizWhiz_k12_v2"
EXTRA_ARGS=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --load_in_8bit)
            EXTRA_ARGS="$EXTRA_ARGS --load_in_8bit"
            shift
            ;;
        --load_in_4bit)
            EXTRA_ARGS="$EXTRA_ARGS --load_in_4bit"
            shift
            ;;
        --cpu)
            EXTRA_ARGS="$EXTRA_ARGS --device cpu"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--model_path PATH] [--load_in_8bit] [--load_in_4bit] [--cpu]"
            exit 1
            ;;
    esac
done

echo "================================"
echo "QuizWhiz_k12_v2 多轮对话系统"
echo "================================"
echo "模型路径: $MODEL_PATH"
echo "参数: $EXTRA_ARGS"
echo "================================"
echo ""

python inference.py --model_path "$MODEL_PATH" $EXTRA_ARGS
