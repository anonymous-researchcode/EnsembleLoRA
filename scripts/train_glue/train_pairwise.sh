task_names=("cb" "rte" "copa" "wic" "wsc.fixed" "boolq") # "multirc"

length=${#task_names[@]}

for ((i = 0; i < $length; i++)); do
  python custom_train_glue_mtl.py --task_names "${task_names[$i]}"\
        --model_key  "meta-llama/Llama-3.1-8B"\
        --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
        --train_lora --lora_rank 16 --lora_alpha 128 \
        --save_name pairwise --epochs 10 --write_results
        
  for ((j = i + 1; j < $length; j++)); do
    python custom_train_glue_mtl.py --task_names "${task_names[$i]}" "${task_names[$j]}" \
        --model_key  "meta-llama/Llama-3.1-8B"\
        --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
        --train_lora --lora_rank 16 --lora_alpha 128 \
        --save_name pairwise --epochs 10 --write_results

    python custom_train_glue_mtl.py --task_names  "${task_names[$i]}" "${task_names[$j]}" \
        --model_key  "meta-llama/Llama-3.1-8B"\
        --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
        --train_lora --lora_rank 16 --lora_alpha 128 \
        --save_name pairwise_4bit --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 

    python custom_train_glue_mtl.py --task_names "${task_names[$i]}" "${task_names[$j]}" \
        --model_key  "meta-llama/Llama-3.1-8B"\
        --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
        --train_lora --lora_rank 16 --lora_alpha 128 \
        --save_name pairwise_3bit --epochs 10 --use_3bit --optimizer "paged_adamw_32bit" --precision "32"
  done
done