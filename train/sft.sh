echo "Starting SWiRL SFT..."
export PYTHONUNBUFFERED=1

# Do all the research.
export WANDB_API_KEY="d8d9e125428dc4ae19cfe692ce098eee9387b5cb" # set this to your own api key
export HUGGINGFACE_TOKEN="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" 
export TRAINING_ID="001"
export WANDB_LOG_MODEL=true

huggingface-cli login --token $HUGGINGFACE_TOKEN

# export CUDA_VISIBLE_DEVICES=""  # if only using some GPUs, specify GPU IDs

# Llama-2 Chat
export OUTPUT_DIR="./swag_models/llama2/" # change this to your own output directory
torchrun --nproc_per_node=8 sft.py \
        --model_name_or_path "meta-llama/Llama-2-7b-chat-hf"
        --bf16 True \
        --model_max_length 32768 \
        --use_flash_attn True \
        --low_rank_training True \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 64 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2650 \
        --save_total_limit 10 \
        --learning_rate 3e-5 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --warmup_steps 30 \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 1 \
        --logging_strategy "steps" \
        --output_dir $OUTPUT_DIR \
        --tf32 True \
        --max_steps 5300 \
        --ddp_find_unused_parameters False


export OUTPUT_DIR="./swag_models/mistral/" # change this to your own output directory
torchrun --nproc_per_node=8 sft.py \
        --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.1"
        --bf16 True \
        --model_max_length 32768 \
        --use_flash_attn True \
        --low_rank_training True \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 64 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2650 \
        --save_total_limit 10 \
        --learning_rate 3e-5 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --warmup_steps 30 \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 1 \
        --logging_strategy "steps" \
        --output_dir $OUTPUT_DIR \
        --tf32 True \
        --max_steps 5300 \
        --ddp_find_unused_parameters False

export OUTPUT_DIR="./swag_models/mistral/" # change this to your own output directory
torchrun --nproc_per_node=8 sft.py \
        --model_name_or_path "meta-llama/Llama-2-7b-chat-hf"
        --bf16 True \
        --model_max_length 32768 \
        --use_flash_attn True \
        --low_rank_training True \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 64 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2650 \
        --save_total_limit 10 \
        --learning_rate 3e-5 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --warmup_steps 30 \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 1 \
        --logging_strategy "steps" \
        --output_dir $OUTPUT_DIR \
        --tf32 True \
        --max_steps 5300 \
        --ddp_find_unused_parameters False

export OUTPUT_DIR="./swag_models/starling/" # change this to your own output directory
torchrun --nproc_per_node=8 sft.py \
        --model_name_or_path "berkeley-nest/Starling-LM-7B-alpha"
        --bf16 True \
        --model_max_length 32768 \
        --use_flash_attn True \
        --low_rank_training True \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 64 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2650 \
        --save_total_limit 10 \
        --learning_rate 3e-5 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --warmup_steps 30 \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 1 \
        --logging_strategy "steps" \
        --output_dir $OUTPUT_DIR \
        --tf32 True \
        --max_steps 5300 \
        --ddp_find_unused_parameters False

export OUTPUT_DIR="./swag_models/deci_lm/" # change this to your own output directory
torchrun --nproc_per_node=8 sft.py \
        --model_name_or_path "Deci/DeciLM-7B-instruct"
        --bf16 True \
        --model_max_length 32768 \
        --use_flash_attn True \
        --low_rank_training True \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 64 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2650 \
        --save_total_limit 10 \
        --learning_rate 3e-5 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --warmup_steps 30 \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 1 \
        --logging_strategy "steps" \
        --output_dir $OUTPUT_DIR \
        --tf32 True \
        --max_steps 5300 \
        --ddp_find_unused_parameters False

# print completion time.
date