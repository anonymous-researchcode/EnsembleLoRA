for task_name in "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" # "winogrande_debiased" "story_cloze" "hellaswag"
do
for rank in 4 16 64 256
do
python custom_train_glue_mtl.py --task_names $task_name\
    --model_key 'meta-llama/Llama-3.1-8B' \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5 \
    --train_lora --lora_rank $rank --lora_alpha $((rank*8)) \
    --save_name varying_rank --epochs 5 --write_results --precision 'bf16-true' \
    --use_qlora --optimizer 'paged_adamw_32bit'
done
done