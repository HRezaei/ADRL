# ADRL — Action Decomposition Reinforcement Learning

Reinforcement learning with large language models (LLMs) in interactive environments.

[[POAD paper @ NeurIPS 2024](https://neurips.cc/virtual/2024/poster/95795)]

## Setup

### Prerequisites
- Conda
- CUDA-capable GPU

### Installation

```bash
conda env create -f environment.yml
conda activate ADRL
```

Or using the lightweight spec:

```bash
conda env create -f env.yaml
conda activate ADRL
pip install -r requirements.txt
```

## Environments

| Script | Runner | Environment |
|--------|--------|-------------|
| `train_virtualhome.py` | `VirtualHomeRunner` | VirtualHome (household tasks) |
| `train_babyai_text.py` | `BabyAITextRunner` | BabyAI-Text (grid-world navigation) |
| `train_overcooked.py` | `OvercookedRunner` | Overcooked (collaborative cooking) |
| `train_datascience.py` | `DataScienceRunner` | Scikit / AutoML pipeline |
| `train_case_study.py` | `CaseStudyRunner` | Case-study benchmark |

## Running

All training scripts are in `mappo/scripts/`. Run from that directory.

### Single GPU

```bash
cd mappo/scripts

# VirtualHome
CUDA_VISIBLE_DEVICES=0 python train_virtualhome.py \
    --env_name "VirtualHome-v1" \
    --algorithm_name "POAD" \
    --experiment_name "test_single" \
    --num_env_steps 6000 \
    --seed 10 \
    --entropy_coef 0.01 \
    --ppo_epoch 1 \
    --num_mini_batch 4 \
    --gradient_cp_steps 8 \
    --model_name "HuggingFaceH4/tiny-random-LlamaForCausalLM"

# Overcooked
CUDA_VISIBLE_DEVICES=0 python train_overcooked.py \
    --env_name "Overcooked-LLMA-v4" \
    --algorithm_name "POAD" \
    --experiment_name "test_single" \
    --num_env_steps 40000 \
    --seed 1 \
    --entropy_coef 0.0001 \
    --gradient_cp_steps 1 \
    --ppo_epoch 5 \
    --num_mini_batch 2 \
    --model_name "HuggingFaceH4/tiny-random-LlamaForCausalLM"

# DataScience
CUDA_VISIBLE_DEVICES=0 python train_datascience.py \
    --env_name "scikit" \
    --algorithm_name "POAD" \
    --experiment_name "test_single" \
    --dataset_name "balance_scale" \
    --flag "balance_scale_poad" \
    --seed 10 \
    --model_name "meta-llama/CodeLlama-7b-hf"

# Case Study
CUDA_VISIBLE_DEVICES=0 python train_case_study.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --experiment_name "case_study" \
    --algorithm_name "POAD" \
    --seed 10 \
    --num_env_steps 50000 \
    --ppo_epoch 1 \
    --num_mini_batch 1

# BabyAI-Text
CUDA_VISIBLE_DEVICES=0 python train_babyai_text.py \
    --env_name "BabyAI-MixedTrainLocal-v0" \
    --algorithm_name "POAD" \
    --experiment_name "test_single" \
    --num_env_steps 6000 \
    --seed 10 \
    --entropy_coef 0.01 \
    --ppo_epoch 1 \
    --num_mini_batch 2 \
    --gradient_cp_steps 16 \
    --model_name "HuggingFaceH4/tiny-random-LlamaForCausalLM"
```

### Multi-GPU (distributed)

Uses `torchrun` with FSDP. The number of rollout threads is divided evenly across GPUs.

```bash
# 2 GPUs — VirtualHome
torchrun --nproc_per_node=2 train_virtualhome.py \
    --env_name "VirtualHome-v1" \
    --algorithm_name "POAD" \
    --experiment_name "test_2gpu" \
    --num_env_steps 6000 \
    --seed 10 \
    --entropy_coef 0.01 \
    --ppo_epoch 1 \
    --num_mini_batch 4 \
    --gradient_cp_steps 8 \
    --model_name "HuggingFaceH4/tiny-random-LlamaForCausalLM"

# 4 GPUs — BabyAI-Text
torchrun --nproc_per_node=4 train_babyai_text.py \
    --env_name "BabyAI-MixedTrainLocal-v0" \
    --algorithm_name "POAD" \
    --experiment_name "test_4gpu" \
    --num_env_steps 6000 \
    --seed 10 \
    --entropy_coef 0.01 \
    --ppo_epoch 1 \
    --num_mini_batch 2 \
    --gradient_cp_steps 16 \
    --model_name "HuggingFaceH4/tiny-random-LlamaForCausalLM"
```

**Note:** For smaller models that fit on one GPU, the default `NO_SHARD` strategy (like DDP) performs better than `FULL_SHARD`. Use `--sharding_strategy full_shard` to enable full parameter sharding for larger models that require splitting across GPUs.

### Common Arguments

#### Setup & Run

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--algorithm_name` | str | `POAD` | Algorithm: `POAD`, `NTPO`, `TWOSOME`, `ARCHER` |
| `--experiment_name` | str | `check` | Experiment identifier |
| `--seed` | int | `1` | Random seed for numpy/torch |
| `--cuda` | bool | `True` | Use GPU (pass `--no-cuda` to disable) |
| `--cuda_deterministic` | bool | `True` | Ensure deterministic CUDA ops (pass `--no-cuda_deterministic` to disable) |
| `--n_training_threads` | int | `16` | Number of torch threads for training |
| `--n_rollout_threads` | int | `32` | Parallel envs for training (divided by world_size in distributed mode) |
| `--n_eval_rollout_threads` | int | `1` | Parallel envs for evaluation |
| `--n_render_rollout_threads` | int | `1` | Parallel envs for rendering |
| `--num_env_steps` | int | `10000000` | Total environment steps to train |
| `--user_name` | str | `xxx` | User name for wandb |
| `--use_wandb` | bool | `False` | Log to wandb (pass `--use_wandb` to enable) |
| `--run_dir_base` | str | `None` | Custom root for results directory |

#### Environment

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--episode_length` | int | `200` | Max episode length |
| `--use_obs_instead_of_state` | bool | `False` | Use local obs instead of global state |

#### Network

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--share_policy` | bool | `True` | Share policy across agents (pass `--no-share_policy` to disable) |
| `--use_centralized_V` | bool | `True` | Centralized critic (pass `--no-use_centralized_V` to disable) |
| `--stacked_frames` | int | `1` | Number of stacked frames |
| `--use_stacked_frames` | bool | `False` | Enable stacked frames |
| `--hidden_size` | int | `64` | Hidden layer dimension |
| `--layer_N` | int | `2` | Number of layers |
| `--use_ReLU` | bool | `True` | Use ReLU (pass `--no-use_ReLU` for Tanh) |
| `--use_popart` | bool | `False` | Use PopArt reward normalization |
| `--use_valuenorm` | bool | `True` | Use running mean/std reward norm (pass `--no-use_valuenorm` to disable) |
| `--use_feature_normalization` | bool | `True` | Apply LayerNorm to inputs (pass `--no-use_feature_normalization` to disable) |
| `--use_orthogonal` | bool | `True` | Orthogonal weight init (pass `--no-use_orthogonal` for xavier uniform) |
| `--gain` | float | `0.01` | Gain for last action layer |

#### Recurrent

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_naive_recurrent_policy` | bool | `False` | Use naive recurrent policy |
| `--use_recurrent_policy` | bool | `False` | Use recurrent policy |
| `--recurrent_N` | int | `1` | Number of recurrent layers |
| `--data_chunk_length` | int | `10` | Chunk length for recurrent training |

#### Optimizer

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lr` | float | `1e-6` | Policy learning rate |
| `--critic_lr` | float | `5e-5` | Critic learning rate |
| `--opti_eps` | float | `1e-5` | Optimizer epsilon |
| `--weight_decay` | float | `0` | Weight decay coefficient |

#### PPO

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--ppo_epoch` | int | `1` | PPO epochs per update |
| `--use_clipped_value_loss` | bool | `True` | Clip value loss (pass `--no-use_clipped_value_loss` to disable) |
| `--clip_param` | float | `0.2` | PPO clip parameter |
| `--num_mini_batch` | int | `4` | Number of mini-batches per PPO epoch |
| `--entropy_coef` | float | `0.01` | Entropy bonus coefficient |
| `--value_loss_coef` | float | `0.5` | Value loss coefficient |
| `--use_max_grad_norm` | bool | `True` | Clip gradients by norm (pass `--no-use_max_grad_norm` to disable) |
| `--max_grad_norm` | float | `0.5` | Max gradient norm |
| `--use_gae` | bool | `True` | Generalized Advantage Estimation (pass `--no-use_gae` to disable) |
| `--gamma` | float | `0.95` | Discount factor |
| `--gae_lambda` | float | `0.95` | GAE lambda parameter |
| `--use_proper_time_limits` | bool | `False` | Account for time limits in return computation |
| `--use_huber_loss` | bool | `False` | Use Huber loss |
| `--use_value_active_masks` | bool | `False` | Mask invalid data in value loss |
| `--use_policy_active_masks` | bool | `True` | Mask invalid data in policy loss (pass `--no-use_policy_active_masks` to disable) |
| `--huber_delta` | float | `10.0` | Huber loss delta |

#### PPG (only for PPG-based algos)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--aux_epoch` | int | `4` | Auxiliary epochs |
| `--clone_coef` | float | `0.01` | Clone term coefficient |

#### LR Schedule

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_linear_lr_decay` | bool | `False` | Linear LR decay |

#### Save / Log

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--save_interval` | int | `100` | Episodes between model saves |
| `--model_save_interval` | int | `5` | Episodes between model saves (alternative) |
| `--log_interval` | int | `1` | Episodes between log prints |
| `--save_json_interval` | int | `1` | Episodes between JSON saves |

#### Evaluation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_eval` | bool | `False` | Enable evaluation during training |
| `--eval_interval` | int | `5` | Episodes between evaluations |
| `--eval_episodes` | int | `16` | Episodes per evaluation |

#### Rendering

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--save_gifs` | int | `0` | Save rendered frames (set to `1` to enable; VirtualHome, BabyAI, Overcooked) |
| `--use_render` | bool | `False` | Render during training |
| `--render_episodes` | int | `5` | Episodes to render |
| `--ifi` | float | `0.1` | Play interval per frame in saved video |

#### Pretrained / Model

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model_dir` | str | `None` | Path to pretrained model |
| `--model_name` | str | `""` | HuggingFace model name or local path |
| `--max_new_tokens` | int | `10` | Max new tokens for LLM generation |
| `--vacab_size` | int | `32000` | Vocabulary size |
| `--gradient_cp_steps` | int | `1` | Gradient checkpointing steps |
| `--use_full_scale` | int | `0` | Use full model weights (no LoRA) |
| `--llm_class_full` | str | `LlamaFullAgent` | Agent class for full-scale LLM |
| `--llm_class_lora` | str | `LlamaLoRAgent` | Agent class for LoRA LLM |
| `--skip_updating_model` | int | `0` | Skip model update (log zero train infos) |
| `--sharding_strategy` | str | `no_shard` | FSDP strategy: `no_shard` or `full_shard` |
| `--push_to_hub_id` | str | `None` | HuggingFace Hub repo id (e.g. `user/repo`). If set, models are pushed to Hub after every local save |
| `--resume_run` | str | `None` | Path to a run directory to resume from (finds latest checkpoint automatically) |
| `--wandb_run_id` | str | `None` | Wandb run id to resume (required with `--resume_run` unless `--force_resume_with_no_wandb` is set) |
| `--force_resume_with_no_wandb` | bool | `False` | Allow resuming without wandb run id |

### HuggingFace Hub Integration

Models can be pushed to HuggingFace Hub alongside local saves. Set `--push_to_hub_id` and provide a `HF_TOKEN`:

```bash
export HF_TOKEN=hf_...  # generate at https://huggingface.co/settings/tokens

python train_virtualhome.py \
    --push_to_hub_id "my-user/my-model" \
    ...
```

Each checkpoint is uploaded as a separate commit named `checkpoint episode <N>`.

### Resume Training

Resume from a crashed or interrupted run by pointing `--resume_run` at the run directory. The latest checkpoint is loaded automatically; model, optimizer, scheduler, RNG, and wandb state are restored.

```bash
python train_virtualhome.py \
    --resume_run "mappo/scripts/results/my_experiment/VirtualHome-v1/TPPO/run0" \
    --wandb_run_id <wandb-run-id> \
    ...
```

Use `--force_resume_with_no_wandb` to skip wandb resume (starts a fresh wandb run instead). The environment seed is advanced by the number of completed episodes so the agent does not replay the same game sequence.

### Script-Specific Arguments

| Script | Flag | Type | Default | Description |
|--------|------|------|---------|-------------|
| `train_virtualhome.py` | `--env_name` | str | `VirtualHome-v1` | `VirtualHome-v1` or `VirtualHome-v2` |
| `train_babyai_text.py` | `--env_name` | str | `BabyAI-Text-v0` | BabyAI-Text environment ID |
| `train_babyai_text.py` | `--num_past_obs` | int | `3` | Past observations in prompt |
| `train_babyai_text.py` | `--use_planner` | int | `0` | Use gold path as teacher actions |
| `train_overcooked.py` | `--env_name` | str | `Overcooked-LLMA-v4` | `Overcooked-LLMA-v4` or `Overcooked-LLMA-v3` |
| `train_overcooked.py` | `--use_planner` | int | `0` | Use planner as teacher |
| `train_datascience.py` | `--env_name` | str | `scikit` | `scikit` or `alfworld` |
| `train_datascience.py` | `--dataset_name` | str | `pharyngitis` | Dataset name (e.g. `balance_scale`, `pharyngitis`) |
| `train_datascience.py` | `--flag` | str | `train` | Run flag for experiment identification |
| `train_datascience.py` | `--split` | bool | `False` | Split dataset |
| `train_case_study.py` | *(no extra args)* | | | Uses only common args |

## Results

### Directory structure

```
mappo/scripts/results/
└── <experiment_name>/
    └── <env_name>/
        └── <algorithm_name>/
            └── run<N>/
                ├── logs/
                │   └── summary.json
                ├── models/
                │   └── episode_<N>/
                │       ├── actor_lora.pth / actor_full.pth
                │       ├── critic_lora.pth / critic_v_head.pth
                │       ├── actor_lora_full_state_dict/  (OPT only)
                │       ├── checkpoint.pt
                │       ├── config.json
                │       └── pytorch_model.bin
                ├── screenshots/
                │   └── env<NN>_ep<NNNN>_run<NN>/
                │       └── step<NNNN>.png
                └── games.json
```

### Monitoring

- **Wandb** (when available): logs config, metrics, and model params.
- **TensorBoard**: logs are written to `logs/` in the run directory.
- **Stdout**: episodic returns and lengths printed during training.

### Key Metrics

- `episodic_return` — cumulative reward per episode
- `success_rate` — fraction of episodes with positive return
- `value_loss` / `policy_loss` — PPO loss components
- `average_step_rewards` — mean reward per step in the buffer
