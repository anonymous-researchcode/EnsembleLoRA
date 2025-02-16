python fast_estimate_linear_regression_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
    --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 2 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name compute_gradients --epochs 0 --precision "bf16-true" \
    --compute_gradient_seed 0 --project_gradients_dim 200 --scale 0.3 --number_of_subsets 120 --subset_size 3

#  "anli_r1" "anli_r2" "anli_r3"
