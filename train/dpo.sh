echo "Starting Forecasting DPO..."
export PYTHONUNBUFFERED=1

# Do all the research.
export WANDB_API_KEY="" # set this to your own api key
export HUGGINGFACE_TOKEN="" # set this to your own api key
export RUN_NAME="forecast-dpo"
export TRAINING_ID="001"
export WANDB_LOG_MODEL=true

huggingface-cli login --token $HUGGINGFACE_TOKEN

# Llama-2 Chat
export OUTPUT_DIR="/lustre/fsw/coreai_dlalgo_llm/zeeshanp/testing/stat165_final_proj/models/dpo/llama2_lr_3e-6_ckpt117/" # change this to your own output directory
torchrun --nproc_per_node=8 /lustre/fsw/coreai_dlalgo_llm/zeeshanp/testing/stat165_final_proj/train/dpo.py \
        --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
        --model_adapter_path "/lustre/fsw/coreai_dlalgo_llm/zeeshanp/testing/stat165_final_proj/models/sft/llama2_lr_3e-6/checkpoint-117/" \
        --cache_dir "/lustre/fsw/coreai_dlalgo_llm/zeeshanp/testing/stat165_final_proj/pretrained_models" \
        --output_dir $OUTPUT_DIR \
        --ddp_find_unused_parameters False

# Llama-3 Chat
export OUTPUT_DIR="/lustre/fsw/coreai_dlalgo_llm/zeeshanp/testing/stat165_final_proj/models/dpo/llama3_lr_3e-6_ckpt117/" # change this to your own output directory
torchrun --nproc_per_node=8 /lustre/fsw/coreai_dlalgo_llm/zeeshanp/testing/stat165_final_proj/train/dpo.py \
        --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
        --model_adapter_path "/lustre/fsw/coreai_dlalgo_llm/zeeshanp/testing/stat165_final_proj/models/sft/llama3_lr_3e-6/checkpoint-117/" \
        --cache_dir "/lustre/fsw/coreai_dlalgo_llm/zeeshanp/testing/stat165_final_proj/pretrained_models" \
        --output_dir $OUTPUT_DIR \
        --ddp_find_unused_parameters False
