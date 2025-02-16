# "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"
python fast_estimate_linear_regression_glue.py --task_names "cb" "rte" \
    --model_key  "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name llama8b_cb_rte_dim_500 --epochs 0 --precision "bf16-true" \
    --compute_gradient_seed 0 --project_gradients_dim 500 --scale 0 --number_of_subsets 120 --subset_size 3