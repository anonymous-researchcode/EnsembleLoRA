python custom_train_glue_mtl.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag" "anli_r1" "anli_r2" "anli_r3"  --model_key "meta-llama/Llama-3.1-8B"\
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 512 --runs 2 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name multitask_16bit --epochs 10 --precision "bf16-true" --accumulate 2

# python custom_train_glue_mtl.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" --model_key "meta-llama/Llama-3.1-8B"\
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 512 --runs 2 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name single_task_4bit --epochs 10 --precision "bf16-true" --accumulate 2 --use_qlora --optimizer "paged_adamw_32bit"
