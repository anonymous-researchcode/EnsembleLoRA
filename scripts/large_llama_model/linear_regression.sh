hf_key="meta-llama/CodeLlama-34b-Instruct-hf"
dev=0
bs=4

python fast_estimate_linear_regression_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
    --model_key $hf_key\
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100\
    --devices $dev --batch_size $bs --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name codellama_glue_10tasks --epochs 0 --precision "bf16-true" \
    --compute_gradient_seed 0 --project_gradients_dim 200 --scale 0.3 --number_of_subsets 120 --subset_size 3


# "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"