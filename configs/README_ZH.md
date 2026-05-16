# 配置文件说明

本目录存放 FINDER 训练使用的 JSON 预设。`train.py` 会先加载对应预设，再应用命令行覆盖项，因此可以把稳定实验设置保存在配置文件里，只在运行时修改少数参数。

## 配置预设

| 文件 | 用途 |
| --- | --- |
| `cn_config.json` | 默认 Critical Node 训练配置。 |
| `cn_cost_config.json` | 带成本的 Critical Node 训练配置。 |
| `nd_config.json` | 默认 Network Dismantling 训练配置。 |
| `nd_cost_config.json` | 带成本的 Network Dismantling 训练配置。 |
| `cn_full_config.json` | 启用 Double DQN、Dueling DQN、PER、Huber loss 和 N-step learning 的 CN 配置。 |
| `finder_defaults.json` | 使用 `base` + `variants` 结构的 fallback / 参考默认配置。 |

`train.py` 通常加载 `cn_config.json`、`cn_cost_config.json`、`nd_config.json` 或 `nd_cost_config.json`。`finder_defaults.json` 是 fallback / reference 文件，不是和各变体 JSON 同形的主预设。

## 顶层字段

| 字段 | 含义 |
| --- | --- |
| `comment` | 配置文件的人类可读备注。 |
| `variant` | 问题变体：`cn`、`cn_cost`、`nd` 或 `nd_cost`。 |
| `device` | 训练设备，例如 `cuda`、`cuda:0` 或 `cpu`。 |
| `base_dir` | 实验输出根目录。 |
| `experiment_name` | `base_dir` 下的实验目录名。 |

## `network` 字段

| 字段 | 含义 |
| --- | --- |
| `min_nodes` / `max_nodes` | 生成训练图时的节点数量范围。 |
| `graph_type` | 图生成器类型，例如 `barabasi_albert` 或 `small_world`；代码也兼容旧写法 `small-world`。 |
| `training_type` | 生成图池时的节点权重赋值策略：`uniform`、`random` 或 `degree`。 |
| `n_train_graphs` | 训练图池中的图数量。 |
| `n_valid_graphs` | 验证 / 评估图池中的图数量。 |

## `model` 字段

| 字段 | 含义 |
| --- | --- |
| `embedding_size` | 图 / 节点隐藏表示维度。 |
| `reg_hidden` | Q 值回归头的隐藏层维度。 |
| `aux_dim` | 辅助图状态特征维度；FINDER 默认使用 4。 |
| `max_bp_iter` | 图消息传递迭代次数。 |
| `aggregator_id` | 邻居聚合方式：`0` 为 sum，`1` 为 mean，`2` 为 GCN 风格。 |
| `embedding_method` | GNN 表示方式选择：`0` 为 structure2vec 风格，`1` 为 GraphSAGE 风格。 |
| `initialization_stddev` | 模型参数初始化标准差。 |

## `training` 字段

| 字段 | 含义 |
| --- | --- |
| `gamma` | 未来奖励折扣因子。 |
| `learning_rate` | 优化器学习率。 |
| `batch_size` | 从 replay memory 采样的 mini-batch 大小。 |
| `memory_size` | replay buffer 容量。 |
| `n_step` | 多步回报的步数。 |
| `max_iterations` | 最大训练迭代数。 |
| `update_target_freq` | target network 更新频率。 |
| `eval_freq` | 训练期间评估频率。 |
| `save_freq` | checkpoint 保存频率。 |
| `graph_pool_update_freq` | 图池刷新或更新频率。 |
| `sampling_freq` | 每隔多少训练迭代采集新完整 episode。 |
| `episodes_per_sampling` | 每次采样阶段采集的 episode 数量。 |
| `eps_start` / `eps_end` | epsilon-greedy 探索率的起始值和最终值。 |
| `eps_decay_steps` | epsilon 衰减步数。 |
| `alpha` | FINDER 图重构 loss 项的权重。 |
| `max_grad_norm` | 可选梯度裁剪阈值；省略时默认使用 `5.0`。 |
| `is_huber_loss` | 是否使用 Huber loss 替代 MSE 计算 TD error。 |
| `is_double_dqn` | 是否启用 Double DQN 目标估计。 |
| `is_dueling_dqn` | 是否启用 dueling value / advantage head。 |
| `is_prioritized_sampling` | 是否启用 prioritized experience replay。 |
| `is_multi_step_dqn` | 是否启用 N-step DQN target。 |
| `priority_epsilon` | PER 中避免零采样概率的小偏移量。 |
| `priority_alpha` | PER 优先级指数。 |
| `priority_beta` | PER importance-sampling 修正因子。 |
| `priority_beta_increment` | 每次采样后 `priority_beta` 的退火增量。 |
| `td_err_upper` | TD error priority 的裁剪上界。 |
| `enable_training_eval` | 是否在训练期间运行评估。 |
| `num_eval_episodes` | 每次评估的 episode 数量。 |
| `num_eval_envs` | 评估时使用的并行环境数量。 |
| `eval_epsilon` | 评估时的探索率；`0.0` 表示纯 greedy。 |

## `vector_env` 字段

| 字段 | 含义 |
| --- | --- |
| `num_envs` | 并行训练环境数量。 |
| `async_env` | 为 true 时使用 `AsyncVectorEnv`，否则使用同步向量环境。 |
| `max_episode_steps` | 可选 episode 截断长度；`null` 表示使用环境默认行为。 |

## 命令行覆盖关系

| 参数 | 对应配置 |
| --- | --- |
| `--variant` | 选择要加载的配置预设。 |
| `--full-tricks` | 为 CN 变体加载 `cn_full_config.json`。 |
| `--seed` | 在构造向量环境前设置进程级随机种子。 |
| `--device` | `device` |
| `--cuda-device` | `device`，形式为 `cuda:<index>`。 |
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
| `--base-dir` | 实验输出根目录。 |
| `--experiment-name` | `base_dir` 下的实验目录名。 |

训练产物会写入 `base_dir / experiment_name`，通常是：

```text
experiments/<experiment_name>/models
experiments/<experiment_name>/logs
```
