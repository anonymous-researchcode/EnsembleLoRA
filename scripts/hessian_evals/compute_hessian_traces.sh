# python compute_hessian_traces.py --task_name "cb" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_cb_lora_r_4_varying_rank_run_0/epoch_epoch=3.pt"

# python compute_hessian_traces.py --task_name "cb" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_cb_lora_r_16_varying_rank_run_0/epoch_epoch=2.pt"

# python compute_hessian_traces.py --task_name "cb" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 64 --lora_alpha 512 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_cb_lora_r_64_varying_rank_run_0/epoch_epoch=4.pt"

# python compute_hessian_traces.py --task_name "cb" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 256 --lora_alpha 2048 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_cb_lora_r_256_varying_rank_run_0/epoch_epoch=3.pt"



# python compute_hessian_traces.py --task_name "rte" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_rte_lora_r_4_varying_rank_run_0/epoch_epoch=4.pt"

# python compute_hessian_traces.py --task_name "rte" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_rte_lora_r_16_varying_rank_run_0/epoch_epoch=1.pt"

python compute_hessian_traces.py --task_name "rte" \
      --model_key "meta-llama/Llama-3.1-8B"\
      --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
      --train_lora --lora_rank 64 --lora_alpha 512 --use_qlora --precision 'bf16-true'\
      --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_rte_lora_r_64_varying_rank_run_0/epoch_epoch=3.pt"

python compute_hessian_traces.py --task_name "rte" \
      --model_key "meta-llama/Llama-3.1-8B"\
      --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
      --train_lora --lora_rank 256 --lora_alpha 2048 --use_qlora --precision 'bf16-true'\
      --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_rte_lora_r_256_varying_rank_run_0/epoch_epoch=0.pt"


python compute_hessian_traces.py --task_name "copa" \
      --model_key "meta-llama/Llama-3.1-8B"\
      --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
      --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora --precision 'bf16-true'\
      --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_copa_lora_r_4_varying_rank_run_0/epoch_epoch=3.pt"

python compute_hessian_traces.py --task_name "copa" \
      --model_key "meta-llama/Llama-3.1-8B"\
      --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
      --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora --precision 'bf16-true'\
      --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_copa_lora_r_16_varying_rank_run_0/epoch_epoch=2.pt"

python compute_hessian_traces.py --task_name "copa" \
      --model_key "meta-llama/Llama-3.1-8B"\
      --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
      --train_lora --lora_rank 64 --lora_alpha 512 --use_qlora --precision 'bf16-true'\
      --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_copa_lora_r_64_varying_rank_run_0/epoch_epoch=1.pt"

python compute_hessian_traces.py --task_name "copa" \
      --model_key "meta-llama/Llama-3.1-8B"\
      --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
      --train_lora --lora_rank 256 --lora_alpha 2048 --use_qlora --precision 'bf16-true'\
      --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_copa_lora_r_256_varying_rank_run_0/epoch_epoch=4.pt"


# python compute_hessian_traces.py --task_name "wic" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_wic_lora_r_4_varying_rank_run_0/epoch_epoch=4.pt"

# python compute_hessian_traces.py --task_name "wic" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_wic_lora_r_16_varying_rank_run_0/epoch_epoch=4.pt"

# python compute_hessian_traces.py --task_name "wic" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 64 --lora_alpha 512 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_wic_lora_r_64_varying_rank_run_0/epoch_epoch=0.pt"

# python compute_hessian_traces.py --task_name "wic" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 256 --lora_alpha 2048 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_wic_lora_r_256_varying_rank_run_0/epoch_epoch=2.pt"


# python compute_hessian_traces.py --task_name "wsc.fixed" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_wsc.fixed_lora_r_4_varying_rank_run_0/epoch_epoch=4.pt"

# python compute_hessian_traces.py --task_name "wsc.fixed" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_wsc.fixed_lora_r_16_varying_rank_run_0/epoch_epoch=4.pt"

# python compute_hessian_traces.py --task_name "wsc.fixed" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 64 --lora_alpha 512 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_wsc.fixed_lora_r_64_varying_rank_run_0/epoch_epoch=2.pt"

# python compute_hessian_traces.py --task_name "wsc.fixed" \
#       --model_key "meta-llama/Llama-3.1-8B"\
#       --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
#       --train_lora --lora_rank 256 --lora_alpha 2048 --use_qlora --precision 'bf16-true'\
#       --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_wsc.fixed_lora_r_256_varying_rank_run_0/epoch_epoch=2.pt"


python compute_hessian_traces.py --task_name "boolq" \
      --model_key "meta-llama/Llama-3.1-8B"\
      --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
      --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora --precision 'bf16-true'\
      --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_boolq_lora_r_4_varying_rank_run_0/epoch_epoch=1.pt"

python compute_hessian_traces.py --task_name "boolq" \
      --model_key "meta-llama/Llama-3.1-8B"\
      --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
      --train_lora --lora_rank 16 --lora_alpha 128 --use_qlora --precision 'bf16-true'\
      --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_boolq_lora_r_16_varying_rank_run_0/epoch_epoch=0.pt"

python compute_hessian_traces.py --task_name "boolq" \
      --model_key "meta-llama/Llama-3.1-8B"\
      --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
      --train_lora --lora_rank 64 --lora_alpha 512 --use_qlora --precision 'bf16-true'\
      --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_boolq_lora_r_64_varying_rank_run_0/epoch_epoch=2.pt"

python compute_hessian_traces.py --task_name "boolq" \
      --model_key "meta-llama/Llama-3.1-8B"\
      --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
      --train_lora --lora_rank 256 --lora_alpha 2048 --use_qlora --precision 'bf16-true'\
      --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_boolq_lora_r_256_varying_rank_run_0/epoch_epoch=0.pt"