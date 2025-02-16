task_names=("cb" "multirc" "rte" "winogrande_debiased" "story_cloze" "hellaswag"  "copa" "wic" "wsc.fixed" "boolq") 
load_model_dir="meta-llama-Llama-3.1-8B_cb_multirc_rte_winogrande_debiased_story_cloze_hellaswag_copa_wic_wsc.fixed_boolq_downsampled_mtl_qadapter_run_0/epoch_epoch=7.pt"

for task_name in "${task_names[@]}"
do
python custom_train_glue_mtl.py --task_names $task_name \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name downsampled_mtft_qlora --epochs 10 --write_results --load_model_dir $load_model_dir
done