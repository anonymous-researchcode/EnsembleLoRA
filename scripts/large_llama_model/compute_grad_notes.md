## notes


when facing this problem while running `compute_gradient.sh`:

```bash


/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py:75: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
Traceback (most recent call last):
  File "/home/michael/project/Quantized-finetuning-code/quantized-finetuning/fast_estimate_compute_gradients_glue.py", line 267, in <module>
    outputs = trainer.predict(lm, dataloaders=data_module.train_dataloader())
  File "/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 858, in predict
    return call._call_and_handle_interrupt(
  File "/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/trainer/call.py", line 47, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 897, in _predict_impl
    results = self._run(model, ckpt_path=ckpt_path)
  File "/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 981, in _run
    results = self._run_stage()
  File "/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1020, in _run_stage
    return self.predict_loop.run()
  File "/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/loops/utilities.py", line 178, in _decorator
    return loop_run(self, *args, **kwargs)
  File "/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/loops/prediction_loop.py", line 104, in run
    self.setup_data()
  File "/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/loops/prediction_loop.py", line 157, in setup_data
    dl = _process_dataloader(trainer, trainer_fn, stage, dl)
  File "/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py", line 484, in _process_dataloader
    dataloader = trainer._data_connector._prepare_dataloader(dataloader, shuffle=is_shuffled, mode=stage)
  File "/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py", line 190, in _prepare_dataloader
    return _update_dataloader(dataloader, sampler, mode=mode)
  File "/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/utilities/data.py", line 134, in _update_dataloader
    dl_args, dl_kwargs = _get_dataloader_init_args_and_kwargs(dataloader, sampler, mode)
  File "/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/utilities/data.py", line 193, in _get_dataloader_init_args_and_kwargs
    dl_kwargs.update(_dataloader_init_kwargs_resolve_sampler(dataloader, sampler, mode))
  File "/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/utilities/data.py", line 283, in _dataloader_init_kwargs_resolve_sampler
    batch_sampler = batch_sampler_cls(
TypeError: MultitaskBatchSampler.__init__() missing 1 required positional argument: 'task_to_datasets'
```

you need to change the raw code of `pytorch_lightning`:

1. click in `/home/michael/anaconda3/envs/adap/lib/python3.10/site-packages/pytorch_lightning/utilities/data.py`

2. located in `def _get_dataloader_init_args_and_kwargs` line 282

3. replace the `try` and following code to the new one below till `except TypeError as ex:`

```python
try:
    if hasattr(batch_sampler, "_task_to_datasets"):
        batch_sampler = batch_sampler_cls(
            sampler,
            batch_size=batch_sampler.batch_size,
            drop_last=(False if is_predicting else batch_sampler.drop_last),
            task_to_datasets=batch_sampler._task_to_datasets, shuffle=False
        )
    else:
        batch_sampler = batch_sampler_cls(
            sampler,
            batch_size=batch_sampler.batch_size,
            drop_last=(False if is_predicting else batch_sampler.drop_last),
        )
```

---
set up enviroments

```bash
mkdir ./sampled_indices
```