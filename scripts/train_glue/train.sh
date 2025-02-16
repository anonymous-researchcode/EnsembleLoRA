# for task_name in "copa" "wsc.fixed" "wic"
# do
# python custom_train_glue.py --task_name $task_name --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name meta_train --epochs 10 

# python custom_train_glue.py --task_name $task_name --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name meta_train --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 
# done

python custom_train_glue.py --task_name "cb" --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name meta_train_3bit --epochs 3 --use_3bit --optimizer "paged_adamw_32bit" --precision "32"

# python custom_train_glue.py --task_name $task_name --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name meta_train --epochs 10 --use_qlora --optimizer "paged_adamw_32bit" --precision "bf16-true" 

# python custom_train_glue.py --task_name $task_name --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name meta_train --epochs 10 --use_3bit --optimizer "paged_adamw_32bit" --precision "32"

# python custom_train_glue.py --task_name $task_name --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name meta_train --epochs 10 --use_2bit --optimizer "paged_adamw_32bit" --precision "32" 
