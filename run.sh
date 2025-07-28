CUDA_VISIBLE_DEVICES=0, python evalute.py --model_name=meta-llama/Llama-3.2-1B-Instruct &
CUDA_VISIBLE_DEVICES=0, python evalute.py --model_name=meta-llama/Llama-3.2-1B-Instruct --cot &

CUDA_VISIBLE_DEVICES=1, python evalute.py --model_name=meta-llama/Llama-3.2-3B-Instruct &
CUDA_VISIBLE_DEVICES=1, python evalute.py --model_name=meta-llama/Llama-3.2-3B-Instruct --cot &

CUDA_VISIBLE_DEVICES=2, python evalute.py --model_name=meta-llama/Llama-3.1-8B-Instruct &
CUDA_VISIBLE_DEVICES=3, python evalute.py --model_name=meta-llama/Llama-3.1-8B-Instruct --cot

CUDA_VISIBLE_DEVICES=1, python evalute.py --model_name=Qwen/Qwen2.5-0.5B-Instruct &
CUDA_VISIBLE_DEVICES=1, python evalute.py --model_name=Qwen/Qwen2.5-0.5B-Instruct --cot &

CUDA_VISIBLE_DEVICES=0, python evalute.py --model_name=Qwen/Qwen2.5-1.5B-Instruct &
CUDA_VISIBLE_DEVICES=0, python evalute.py --model_name=Qwen/Qwen2.5-1.5B-Instruct --cot

CUDA_VISIBLE_DEVICES=1, python evalute.py --model_name=Qwen/Qwen2.5-3B-Instruct &
CUDA_VISIBLE_DEVICES=1, python evalute.py --model_name=Qwen/Qwen2.5-3B-Instruct --cot

CUDA_VISIBLE_DEVICES=2, python evalute.py --model_name=Qwen/Qwen2.5-7B-Instruct &
CUDA_VISIBLE_DEVICES=3, python evalute.py --model_name=Qwen/Qwen2.5-7B-Instruct --cot

CUDA_VISIBLE_DEVICES=0, python evalute.py --model_name=Qwen/Qwen2.5-14B-Instruct &
CUDA_VISIBLE_DEVICES=0, python evalute.py --model_name=Qwen/Qwen2.5-14B-Instruct --cot

CUDA_VISIBLE_DEVICES=1, python evalute.py --model_name=google/gemma-2-2b-it &
CUDA_VISIBLE_DEVICES=1, python evalute.py --model_name=google/gemma-2-2b-it --cot

CUDA_VISIBLE_DEVICES=3, python evalute.py --model_name=google/gemma-2-9b-it &
CUDA_VISIBLE_DEVICES=3, python evalute.py --model_name=google/gemma-2-9b-it --cot
