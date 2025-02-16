# python fast_estimate_compute_gradients_glue.py --task_names "cb" "rte"\
#     --model_key "meta-llama/Llama-3.1-8B"\
#     --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 512 --runs 1 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --use_qlora --optimizer "paged_adamw_32bit"\
#     --save_name llama8b_cb_rte_dim_200 --epochs 0 --precision "bf16-true" \
#     --compute_gradient_steps 10000000 --compute_gradient_seed 0 --project_gradients_dim 200

# python fast_estimate_compute_gradients_glue.py --task_names "cb" "rte"\
#     --model_key "meta-llama/Llama-3.1-8B"\
#     --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 512 --runs 1 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --use_qlora --optimizer "paged_adamw_32bit"\
#     --save_name llama8b_cb_rte_dim_500 --epochs 0 --precision "bf16-true" \
#     --compute_gradient_steps 10000000 --compute_gradient_seed 0 --project_gradients_dim 500
    
python fast_estimate_compute_gradients_glue.py --task_names "cb" "rte"\
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 512 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name llama8b_cb_rte_dim_1000 --epochs 0 --precision "bf16-true" \
    --compute_gradient_steps 10000000 --compute_gradient_seed 0 --project_gradients_dim 800
# --start_step 9581
#  "winogrande_debiased" "story_cloze" "hellaswag" "anli_r1" "anli_r2" "anli_r3"
#     --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100\
# "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"