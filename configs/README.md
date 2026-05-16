# Configuration Reference

This directory contains reusable JSON presets for FINDER training. `train.py` loads one preset first, then applies CLI overrides so experiments can share stable config files while changing only a few run-time knobs.

## Presets

| File | Purpose |
| --- | --- |
| `cn_config.json` | Default Critical Node training setup. |
| `cn_cost_config.json` | Cost-aware Critical Node training setup. |
| `nd_config.json` | Default Network Dismantling training setup. |
| `nd_cost_config.json` | Cost-aware Network Dismantling training setup. |
| `cn_full_config.json` | CN setup with Double DQN, Dueling DQN, PER, Huber loss, and N-step learning. |
| `finder_defaults.json` | Fallback/reference defaults using a `base` + `variants` structure. |

`train.py` normally loads `cn_config.json`, `cn_cost_config.json`, `nd_config.json`, or `nd_cost_config.json`. `finder_defaults.json` is a fallback/reference file and is not shaped like the per-variant presets.

## Top-Level Fields

| Field | Meaning |
| --- | --- |
| `comment` | Human-readable note for the config file. |
| `variant` | Problem variant: `cn`, `cn_cost`, `nd`, or `nd_cost`. |
| `device` | Training device, such as `cuda`, `cuda:0`, or `cpu`. |
| `base_dir` | Root directory for experiment outputs. |
| `experiment_name` | Experiment folder name under `base_dir`. |

## `network` Fields

| Field | Meaning |
| --- | --- |
| `min_nodes` / `max_nodes` | Node-count range for generated training graphs. |
| `graph_type` | Graph generator type, for example `barabasi_albert` or `small_world`. The code also accepts the legacy spelling `small-world`. |
| `training_type` | Node weight assignment strategy for generated graph pools: `uniform`, `random`, or `degree`. |
| `n_train_graphs` | Number of graphs in the training graph pool. |
| `n_valid_graphs` | Number of graphs in the validation/evaluation graph pool. |

## `model` Fields

| Field | Meaning |
| --- | --- |
| `embedding_size` | Hidden graph/node embedding dimension. |
| `reg_hidden` | Hidden dimension of the Q-value regression head. |
| `aux_dim` | Auxiliary graph-state feature dimension. FINDER uses 4 by default. |
| `max_bp_iter` | Number of graph message-passing iterations. |
| `aggregator_id` | Neighbor aggregation mode: `0` sum, `1` mean, `2` GCN-style. |
| `embedding_method` | GNN embedding method selector: `0` structure2vec-style, `1` GraphSAGE-style. |
| `initialization_stddev` | Standard deviation for model parameter initialization. |

## `training` Fields

| Field | Meaning |
| --- | --- |
| `gamma` | Discount factor for future rewards. |
| `learning_rate` | Optimizer learning rate. |
| `batch_size` | Mini-batch size sampled from replay memory. |
| `memory_size` | Replay buffer capacity. |
| `n_step` | Number of steps used by multi-step returns. |
| `max_iterations` | Maximum number of training iterations. |
| `update_target_freq` | Target-network update frequency. |
| `eval_freq` | Training-time evaluation frequency. |
| `save_freq` | Checkpoint saving frequency. |
| `graph_pool_update_freq` | Frequency for refreshing or updating graph pools. |
| `sampling_freq` | How often the trainer collects new complete episodes. |
| `episodes_per_sampling` | Number of episodes collected per sampling phase. |
| `eps_start` / `eps_end` | Initial and final epsilon for epsilon-greedy exploration. |
| `eps_decay_steps` | Number of steps used for epsilon decay. |
| `alpha` | Weight of the FINDER graph reconstruction loss term. |
| `max_grad_norm` | Optional gradient clipping threshold; defaults to `5.0` when omitted. |
| `is_huber_loss` | Use Huber loss instead of MSE for TD errors. |
| `is_double_dqn` | Enable Double DQN target estimation. |
| `is_dueling_dqn` | Enable dueling value/advantage heads. |
| `is_prioritized_sampling` | Enable prioritized experience replay. |
| `is_multi_step_dqn` | Enable N-step DQN targets. |
| `priority_epsilon` | Small priority offset to avoid zero sampling probability. |
| `priority_alpha` | PER priority exponent. |
| `priority_beta` | PER importance-sampling correction factor. |
| `priority_beta_increment` | Per-sampling increment for annealing `priority_beta`. |
| `td_err_upper` | Upper bound used when clipping TD error priorities. |
| `enable_training_eval` | Whether to run evaluation during training. |
| `num_eval_episodes` | Number of episodes per evaluation. |
| `num_eval_envs` | Number of parallel environments used for evaluation. |
| `eval_epsilon` | Exploration rate during evaluation; `0.0` means greedy. |

## `vector_env` Fields

| Field | Meaning |
| --- | --- |
| `num_envs` | Number of parallel training environments. |
| `async_env` | Use `AsyncVectorEnv` when true, otherwise synchronous vector execution. |
| `max_episode_steps` | Optional episode truncation length; `null` keeps the environment default. |

## CLI Override Mapping

| Argument | Config field |
| --- | --- |
| `--variant` | Selects which preset is loaded. |
| `--full-tricks` | Loads `cn_full_config.json` for the CN variant. |
| `--seed` | Sets process-level random seeds before vector environments are constructed. |
| `--device` | `device` |
| `--cuda-device` | `device`, using `cuda:<index>`. |
| `--max-iterations` | `training.max_iterations` |
| `--batch-size` | `training.batch_size` |
| `--learning-rate` | `training.learning_rate` |
| `--num-envs` | `vector_env.num_envs` |
| `--sync-env` | `vector_env.async_env = false` |
| `--eval-freq` | `training.eval_freq` |
| `--save-freq` | `training.save_freq` |
| `--eval-episodes` | `training.num_eval_episodes` |
| `--eval-envs` | `training.num_eval_envs` |
| `--no-eval` | `training.enable_training_eval = false` |
| `--max-grad-norm` | `training.max_grad_norm` |
| `--base-dir` | Experiment output root directory. |
| `--experiment-name` | Experiment folder name under `base_dir`. |

Training artifacts are written to `base_dir / experiment_name`, typically:

```text
experiments/<experiment_name>/models
experiments/<experiment_name>/logs
```
