# EnsembleLoRA



## Setup the environment

```
pip install -r requirements.txt
python setup.py develop
```

### Environment

The code has been tested on Python<=3.10, PyTorch Lightning<=1.9, PyTorch>=2.0



## Running scripts

- Use `custom_train_glue_mtl.py` to conduct fine-tuning on multiple NLP datasets. Please see the examples in the following scripts. 

```
python custom_train_glue_mtl.py --task_names $task_name\
    --model_key 'meta-llama/Llama-3.1-8B' \
    --devices 2 --batch_size 4 --inference_batch_size 4 --max_length 512 --runs 1 --lr 5e-5 \
    --train_lora --lora_rank 128 --lora_alpha 1024 --use_qlora --optimizer 'paged_adamw_32bit'\
    --save_name $save_name --epochs 5 --write_results --precision 'bf16-true' 
    
python custom_train_glue_mtl.py --task_names $task_name \
    --model_key  "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 4 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
    --train_adapter --reduction_factor 128 --use_qadapter --optimizer 'paged_adamw_32bit'\
    --save_name $save_name --epochs 10 --write_results --precision 'bf16-true' 
```

- Use `fast_estimate_compute_gradients_glue.py` to evaluate the gradients for the first-order approximation.

```
python fast_estimate_compute_gradients_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name llama8b_10_tasks_dim_400 --epochs 0 --precision "bf16-true" \
    --compute_gradient_steps 10000000 --compute_gradient_seed 0 --project_gradients_dim 400
```

- Use `fast_estimate_linear_regression_glue.py` to estimate the approximated fine-tuning performances

```
ython fast_estimate_linear_regression_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag" \
    --model_key  "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name llama8b_10_tasks_dim_400 --epochs 0 --precision "bf16-true" \
    --compute_gradient_seed 0 --project_gradients_dim 400 \
    --regularization_lambda $lambda --load_sample_task_dir "llama8b_glue_10tasks_meta-llama-Llama-3.1-8B_lora_r_16_size_3.0_scale_0.3" --downsample $downsample
```

- Use `compute_hessian_traces.py` to evaluate the Hessian of fine-tuned models, for example: 

```
python compute_hessian_traces.py --task_name "cb" \
      --model_key "meta-llama/Llama-3.1-8B"\
      --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
      --train_lora --lora_rank 16 --lora_alpha 128 --precision 'bf16-true'\
      --save_name test_hessian --num_samples 100 

python compute_hessian_traces.py --task_name "cb" \
      --model_key "meta-llama/Llama-3.1-8B"\
      --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
      --train_adapter --reduction_factor 128 --use_qadapter --precision 'bf16-true'\
      --save_name test_hessian --num_samples 100 
```

- Use `clustering.py` to run examples of our clustering algorithms. 
- Use `measure_memory.py` to evaluate the memory cost of the ensemble models. 

