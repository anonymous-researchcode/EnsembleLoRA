hf_key="meta-llama/CodeLlama-34b-Instruct-hf"
dev=0
bs=4
sn=codellama_glue_10tasks

python fast_estimate_compute_gradients_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
    --model_key $hf_key\
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100\
    --devices $dev --batch_size $bs --inference_batch_size 4 --max_length 256 --runs 1 --lr 5e-5\
    --save_name $sn --epochs 0 --precision "bf16-true" \
    --compute_gradient_steps 10000000 --compute_gradient_seed 0 --project_gradients_dim 200 --start_step 0\
    --train_lora --use_qlora --lora_rank 16 --lora_alpha 128 \
    
    
# --start_step 9581
#  "winogrande_debiased" "story_cloze" "hellaswag" "anli_r1" "anli_r2" "anli_r3"
# "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"