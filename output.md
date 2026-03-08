==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 64 | Num Epochs = 1 | Total steps = 20
O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 1
\        /    Data Parallel GPUs = 1 | Total batch size (2 x 1 x 1) = 2
 "-____-"     Trainable parameters = 66,060,288 of 4,088,528,384 (1.62% trained)
WARNING 03-08 19:58:35 [input_processor.py:168] vLLM has deprecated support for supporting different tokenizers for different LoRAs. By default, vLLM uses base model's tokenizer. If you are using a LoRA with its own tokenizer, consider specifying `--tokenizer [lora_path]` to use the LoRA tokenizer.
/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py:908: DeprecationWarning: 
    Prefer using device seq_lens directly to avoid implicit H<>D sync.
    If a CPU copy is needed, use `seq_lens.cpu()` instead.
    Will be removed in a future release, please migrate as soon as possible.
    
  seq_lens_cpu = common_attn_metadata.seq_lens_cpu if needs_seq_lens_cpu else None
/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py:908: DeprecationWarning: 
    Prefer using device seq_lens directly to avoid implicit H<>D sync.
    If a CPU copy is needed, use `seq_lens.cpu()` instead.
    Will be removed in a future release, please migrate as soon as possible.
    
  seq_lens_cpu = common_attn_metadata.seq_lens_cpu if needs_seq_lens_cpu else None
/usr/local/lib/python3.12/dist-packages/jupyter_client/session.py:203: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  return datetime.utcnow().replace(tzinfo=utc)
/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py:908: DeprecationWarning: 
    Prefer using device seq_lens directly to avoid implicit H<>D sync.
    If a CPU copy is needed, use `seq_lens.cpu()` instead.
    Will be removed in a future release, please migrate as soon as possible.
    
  seq_lens_cpu = common_attn_metadata.seq_lens_cpu if needs_seq_lens_cpu else None
/usr/local/lib/python3.12/dist-packages/jupyter_client/session.py:203: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  return datetime.utcnow().replace(tzinfo=utc)
Unsloth: Will smartly offload gradients to save VRAM!
/usr/local/lib/python3.12/dist-packages/jupyter_client/session.py:203: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  return datetime.utcnow().replace(tzinfo=utc)
 [20/20 05:11, Epoch 0/1]
Step	Training Loss	reward	reward_std	completions / mean_length	completions / min_length	completions / max_length	completions / clipped_ratio	completions / mean_terminated_length	completions / min_terminated_length	completions / max_terminated_length	sampling / sampling_logp_difference / mean	sampling / sampling_logp_difference / max	sampling / importance_sampling_ratio / min	sampling / importance_sampling_ratio / mean	sampling / importance_sampling_ratio / max	kl	rewards / reward_env_return / mean	rewards / reward_env_return / std	rewards / reward_valid_action / mean	rewards / reward_valid_action / std	rewards / reward_job_completion / mean	rewards / reward_job_completion / std
1	0.000000	0.000000	0.000000	142.000000	92.000000	192.000000	0.500000	92.000000	92.000000	92.000000	0	0	0	0	0	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
2	-0.000000	0.000000	0.000000	192.000000	192.000000	192.000000	1.000000	0.000000	0.000000	0.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
3	-0.000000	0.000000	0.000000	192.000000	192.000000	192.000000	1.000000	0.000000	0.000000	0.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
4	-0.000000	0.000000	0.000000	192.000000	192.000000	192.000000	1.000000	0.000000	0.000000	0.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
5	-0.000000	0.000000	0.000000	120.500000	49.000000	192.000000	0.500000	49.000000	49.000000	49.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
6	-0.000000	0.000000	0.000000	192.000000	192.000000	192.000000	1.000000	0.000000	0.000000	0.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
7	-0.000000	0.000000	0.000000	192.000000	192.000000	192.000000	1.000000	0.000000	0.000000	0.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
8	-0.000000	0.000000	0.000000	160.000000	128.000000	192.000000	0.500000	128.000000	128.000000	128.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
9	0.000000	0.000000	0.000000	192.000000	192.000000	192.000000	1.000000	0.000000	0.000000	0.000000	No Log	No Log	No Log	No Log	No Log	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
10	-0.000000	0.000000	0.000000	192.000000	192.000000	192.000000	1.000000	0.000000	0.000000	0.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
11	0.000000	0.000000	0.000000	192.000000	192.000000	192.000000	1.000000	0.000000	0.000000	0.000000	No Log	No Log	No Log	No Log	No Log	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
12	0.000000	0.000000	0.000000	191.000000	190.000000	192.000000	0.500000	190.000000	190.000000	190.000000	No Log	No Log	No Log	No Log	No Log	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
13	0.000000	0.000000	0.000000	131.500000	71.000000	192.000000	0.500000	71.000000	71.000000	71.000000	No Log	No Log	No Log	No Log	No Log	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
14	-0.000000	0.000000	0.000000	190.000000	188.000000	192.000000	0.500000	188.000000	188.000000	188.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
15	-0.000000	0.000000	0.000000	192.000000	192.000000	192.000000	1.000000	0.000000	0.000000	0.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
16	-0.000000	0.000000	0.000000	192.000000	192.000000	192.000000	1.000000	0.000000	0.000000	0.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
17	-0.000000	0.000000	0.000000	192.000000	192.000000	192.000000	1.000000	0.000000	0.000000	0.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
18	-0.000000	0.000000	0.000000	122.000000	52.000000	192.000000	0.500000	52.000000	52.000000	52.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
19	-0.000000	0.000000	0.000000	192.000000	192.000000	192.000000	1.000000	0.000000	0.000000	0.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
20	-0.000000	0.000000	0.000000	192.000000	192.000000	192.000000	1.000000	0.000000	0.000000	0.000000	No Log	No Log	No Log	No Log	No Log	-0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
/usr/local/lib/python3.12/dist-packages/jupyter_client/session.py:203: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  return datetime.utcnow().replace(tzinfo=utc)
/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py:908: DeprecationWarning: 
    Prefer using device seq_lens directly to avoid implicit H<>D sync.
    If a CPU copy is needed, use `seq_lens.cpu()` instead.
    Will be removed in a future release, please migrate as soon as possible.
    
  seq_lens_cpu = common_attn_metadata.seq_lens_cpu if needs_seq_lens_cpu else None
/usr/local/lib/python3.12/dist-packages/jupyter_client/session.py:203: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  return datetime.utcnow().replace(tzinfo=utc)
TrainOutput(global_step=20, training_loss=-2.682549142670951e-13, metrics={'train_runtime': 381.8669, 'train_samples_per_second': 0.105, 'train_steps_per_second': 0.052, 'total_flos': 0.0, 'train_loss': -2.682549142670951e-13})