task_names=("rte")
length=${#task_names[@]}

for ((j = 0; j < $length; j++)); do
  python custom_train_glue_mtl_seq_bn.py --task_names "${task_names[$j]}" \
      --model_key  "meta-llama/Llama-3.2-1B"\
      --devices 1 --batch_size 4 --inference_batch_size 8 --max_length 256 --runs 2 --lr 5e-5\
      --train_adapter --lora_rank 16 --lora_alpha 128\
      --save_name pairwise_adapter --epochs 10 --write_results