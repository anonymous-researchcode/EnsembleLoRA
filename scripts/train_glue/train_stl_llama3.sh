# python evaluate_merged_model.py --task_names "copa" "cb" "rte" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 10\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name merging_tinyllama --precision "bf16-true" \
#     --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt" \
#     "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_boolq_lora_r_16_boost_4bit_run_0/epoch_epoch=9.pt"\
#     "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt"\
#     --merge_strategy "simple_ensemble" --use_qlora --optimizer "paged_adamw_32bit"

# python evaluate_merged_model.py --task_names "copa" "cb" "rte" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 10\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name merging_tinyllama --precision "bf16-true" \
#     --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_1/epoch_epoch=9.pt" \
#     "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_boolq_lora_r_16_boost_4bit_run_1/epoch_epoch=9.pt"\
#     "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_1/epoch_epoch=2.pt"\
#     --merge_strategy "simple_ensemble" --use_qlora --optimizer "paged_adamw_32bit"

# python evaluate_merged_model.py --task_names "copa" "cb" "rte" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 10\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name merging_tinyllama --precision "bf16-true" \
#     --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt" \
#     "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_boolq_lora_r_16_boost_4bit_run_0/epoch_epoch=9.pt"\
#     "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_0/epoch_epoch=6.pt"\
#     --merge_strategy "max_ensemble" --use_qlora --optimizer "paged_adamw_32bit"


# python evaluate_merged_model.py --task_names "copa" "cb" "rte" "wic" "wsc.fixed" "boolq" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 10\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name merging_tinyllama --precision "bf16-true" \
#     --merge_model_dirs "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_rte_copa_wsc.fixed_lora_r_16_boost_4bit_run_1/epoch_epoch=9.pt" \
#     "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_cb_boolq_lora_r_16_boost_4bit_run_1/epoch_epoch=9.pt"\
#     "TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_wic_lora_r_16_boost_4bit_run_1/epoch_epoch=2.pt"\
#     --merge_strategy "max_ensemble" --use_qlora --optimizer "paged_adamw_32bit"

for task_name in "winogrande_debiased" "story_cloze" "anli_r1" "anli_r2" "anli_r3" "hellaswag" "multirc" # "cb" "rte" "copa" "wic" "wsc.fixed" "boolq"
do
python custom_train_glue.py --task_name $task_name --model_key "meta-llama/Llama-3.1-8B"\
    --devices 2 3  --batch_size 4 --inference_batch_size 4 --max_length 512 --runs 2 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name single_task_16bit --epochs 10 --precision "bf16-true" --accumulate 2 # --use_qlora --optimizer "paged_adamw_32bit"
done

# python custom_train_glue_mtl.py --task_name "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" --model_key "meta-llama/Llama-3.1-8B"\
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 512 --runs 2 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name single_task_16bit --epochs 10 --precision "bf16-true" --accumulate 2 

# python custom_train_glue_mtl.py --task_name "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" --model_key "meta-llama/Llama-3.1-8B"\
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 512 --runs 2 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name single_task_4bit --epochs 10 --precision "bf16-true" --accumulate 2 --use_qlora --optimizer "paged_adamw_32bit"
