rdzv_ip=$1
wandb login xxx
torchrun --nnodes=4 --nproc_per_node=4 --rdzv-id=376 --rdzv-backend=c10d \
 --rdzv-endpoint="${rdzv_ip}:29500" \
 src/marvin_recipes/scripts/fsdp_finetune.py --model_chk="/data/llama-2-7b-hf" \
 --remote_dataset="s3://some_bucket/nlu_instruct/recogs_pairs_2" \
 --run_name="fsdp_recogs_llama2_7b_mb2" --num_epochs=4 \
 --mini_batch_size=2 \
 --job_checkpoint_dir="/data/jobs/partial_checkpoints" \
 --save_dir="/data/jobs/model_checkpoints" \
 --checkpoint_steps=120 \
 --warmup_steps=256 \
 --optimizer="anyadamw" \
 --data_loader_workers=1 \
 --grad_acc_steps=4 \
 --external_logger="wandb" --log_project_name="recogs-llm" --local_data_path="data_tmp"
