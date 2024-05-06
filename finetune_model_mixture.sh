# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=""

# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
deepspeed --num_gpus 4 --master_port=9901 src/train_bash.py \
    --run_name finetune_mlprouter_ffn_mixture \
    --report_to wandb \
    --deepspeed ds_config/config.json \
    --stage sft \
    --do_train \
    --train_router \
    --model_name_or_path /data/data2/llm_mixture_outputs_new/llama2_2x7b_vicuna_ffn_mixture \
    --overwrite_output_dir \
    --dataset gpt4all \
    --cache_path data_cache/gpt4all \
    --template default \
    --finetuning_type full \
    --output_dir /data/data2/llm_mixture_outputs_new/llama2_2x7b_vicuna_ffn_mixture_sft \
    --overwrite_cache \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --save_steps 9000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16