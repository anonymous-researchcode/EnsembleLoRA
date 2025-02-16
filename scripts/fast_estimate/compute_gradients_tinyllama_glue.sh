python fast_estimate_compute_gradients_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag" "anli_r1" "anli_r2" "anli_r3"\
    --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 512 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name compute_gradients --epochs 0 --precision "bf16-true" \
    --compute_gradient_steps 10000000 --compute_gradient_seed 0 --project_gradients_dim 200 --start_step 9581


#  "winogrande_debiased" "story_cloze" "hellaswag" "anli_r1" "anli_r2" "anli_r3"