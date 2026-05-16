[English](README.md) | 中文

# FINDER 环境

本目录包含基于图的优化问题的FINDER深度强化学习环境的Python实现。这些环境从原始的Cython/C++实现转换为使用NetworkX和gymnasium的纯Python版本。

## 概述

FINDER框架包含两个主要问题的四个环境变体：

### 关键节点(CN)问题
- **目标**：最小化连通分量分解(CND)分数
- **CND分数**：对所有连通分量求和 |C| × (|C| - 1) / 2
- **环境**：
  - `CriticalNodeEnv`：无权重版本
  - `CriticalNodeCostEnv`：有权重版本（考虑节点移除成本）

### 网络拆解(ND)问题
- **目标**：最小化最大连通分量的大小
- **目的**：通过移除战略性节点来分割网络
- **环境**：
  - `NetworkDismantlingEnv`：无权重版本
  - `NetworkDismantlingCostEnv`：有权重版本（考虑节点移除成本）

## 文件

- `base_env.py`：带通用功能的抽象基类
- `cn_env.py`：关键节点环境（无权重）
- `cn_cost_env.py`：带成本的关键节点环境（有权重）
- `nd_env.py`：网络拆解环境（无权重）
- `nd_cost_env.py`：带成本的网络拆解环境（有权重）
- `__init__.py`：带工厂函数的包初始化（`make_env`/`make_batch_env`；批处理基于 gymnasium.vector）

## 快速开始

```python
from envs import make_env, make_batch_env
import networkx as nx

# 创建单个环境
env = make_env('cn', max_nodes=50, seed=42)

# 创建自己的图或让环境生成一个
graph = nx.barabasi_albert_graph(30, 3)

# 使用自定义图重置环境
obs, info = env.reset(options={'graph': graph})

# 执行动作（移除节点）
action = env.random_action()  # 或使用你的RL智能体
obs, reward, terminated, truncated, info = env.step(action)

# 渲染当前状态
env.render()
```

## 环境工厂

使用工厂函数来轻松创建环境：

```python
from envs import make_env, make_batch_env

# 单个环境
cn_env = make_env('cn', max_nodes=50)
cn_cost_env = make_env('cn_cost', max_nodes=50, weight_range=(0.0, 1.0))
nd_env = make_env('nd', max_nodes=50)
nd_cost_env = make_env('nd_cost', max_nodes=50)

# 用于并行训练的批处理环境（内部使用 gymnasium.vector，保持观测为字典）
batch_env = make_batch_env('cn', batch_size=64, max_nodes=50)
```

## 状态空间

- 环境返回 NetworkX 格式的结构化观测（Dict）：
  - `graph`：当前图（节点已物理删除）。节点属性 `features` 在无成本环境中为 `[1.0, 1.0]`，在带成本环境中为 `[weight, 1.0]`。
  - `aux_features`：长度为4的辅助特征，严格对齐原始FINDER：
  1. 覆盖率 `|removed| / n_orig`
  2. 边覆盖率 `( |E_orig| − |E_cur| ) / |E_orig|`
  3. 两跳密度 `Σ_v C(deg_v, 2) / n_orig^2`
  4. 常数 `1.0`
- `step`、`remaining_nodes`、`action_list` 等运行时元信息位于 `info` 字典中，不属于 observation。

## 动作空间

- **语义**：动作是“节点ID”（NetworkX 节点标签），表示要移除的节点
- **占位空间**：为兼容 gymnasium.vector 的空间检查，环境内部将 `action_space`/`observation_space` 设为占位符 `AnySpace(shape=None, dtype=object)`；这不会限制实际返回的字典观测或动作类型
- **有效性**：调用方需保证所选节点存在于当前图中；若动作无效，环境会返回惩罚（如 `-1.0`），并继续当前回合

## 奖励函数

### 关键节点(CN)
```
奖励 = - CND_remaining / [ n^2 (n-1) / 2 ]    （n 为原始节点数）
```

### 关键节点成本(CN_cost)
```
奖励 = - ( maxCC / n ) × ( weight(action) / total_weight )
```

### 网络拆解(ND) 
```
奖励 = - maxCC / n^2
```

### 网络拆解成本(ND_cost)
```
奖励 = - ( maxCC / n ) × ( weight(action) / total_weight )
```

## 自定义图和权重

```python
import random
import networkx as nx

# 创建自定义图
graph = nx.erdos_renyi_graph(40, 0.1)

# 对于成本环境，指定节点权重
node_weights = {node: random.uniform(0.1, 3.0) for node in graph.nodes()}

# 使用自定义图和权重重置
options = {
    'graph': graph,
    'node_weights': node_weights  # 仅对成本环境
}
obs, info = env.reset(options=options)
```

## 批处理（基于 gymnasium.vector）

```python
from envs import make_batch_env

# 创建批处理环境（内部 Async/SyncVectorEnv）
batch_env = make_batch_env('cn', batch_size=32)

# 创建多个图
graphs = [nx.barabasi_albert_graph(35, 3) for _ in range(32)]

# 重置所有环境（obs 为长度=批大小的字典列表；infos 为字典，键的值按批量聚合）
obs_batch, infos = batch_env.reset(graphs=graphs)

# 对所有环境执行步骤（自动对 done 的子环境执行重置）
actions = [0 for _ in range(32)]
obs, rewards, terminated, truncated, infos = batch_env.step(actions)

# 注意：向量环境返回的 infos 是“字典”，其每个键的值都按批量维度聚合（长度=批大小）。
# 当某子环境在该步完成回合时，infos 还会包含 'final_observation' 与 'final_info' 键（同样按批量聚合）。
```

## 观测与占位空间说明（重要）

- 为适配向量化环境，`envs/gym_batch.py` 内部对单环境进行轻量包装，仅替换空间为 `AnySpace`，不改变 `reset/step` 返回的原始字典观测。
- 向量环境返回：
  - `obs`：长度为批大小的“字典观测列表”
  - `infos`：字典（dict-of-arrays），每个键的值为长度=批大小的序列；回合完成时还会出现 `final_observation`/`final_info` 两组键。
- 成本变体的“节点特征顺序”为 `[weight, 1.0]`（无权重环境中 `weight=1.0`）。

## 兼容性

- **gymnasium**：完全支持gymnasium.Env接口
- **NetworkX**：使用NetworkX进行图操作
- **NumPy**：观测和奖励使用NumPy数组
- **向量化**：用于并行处理的批处理环境（gymnasium.vector）

## 与RL框架集成

FINDER 观测中包含 NetworkX 图对象，因此应使用本仓库提供的批处理适配器，不要直接构造 `gymnasium.vector.AsyncVectorEnv`：

```python
from envs import make_batch_env
import networkx as nx

vec_env = make_batch_env('cn', batch_size=4, async_env=True)
```

## 性能说明

- NetworkX操作对于多达约1000个节点的图来说是相当快的
- 批处理环境并行处理多个图
- 状态计算在可能的地方使用缓存进行优化
- 使用较小的图（30-50个节点）以获得更快的训练速度
- 对于更大的图，考虑优化连通性计算

## API参考

### 基础环境参数
- `max_nodes`：图的最大节点数（默认：51）
- `min_nodes`：图的最小节点数（默认：30）
- `aux_dim`：辅助特征维度（默认：4）
- `seed`：随机种子（可选）

### 终止条件说明
- 原始FINDER：无固定“最大移除数”限制，终止基于各问题的图覆盖 / 拆解条件。
- 当前实现：默认禁用步数截断（`use_step_truncation = False`）。只有显式启用时，才使用 `max_steps = max_nodes` 截断。

### 成本环境额外参数
- `weight_range`：随机节点权重范围（默认：(0.0, 1.0)，与原始FINDER一致）

### 主要方法
- `reset()`：重置环境到初始状态
- `step(action)`：执行动作并返回新状态
- `render()`：显示当前环境状态
- `random_action()`：选择随机有效动作
- `betweenness_action()`：基于中介中心性选择动作

### 环境特定方法
- `get_robustness(solution)`：计算解决方案的鲁棒性度量
- `get_remaining_cnd_score()`：获取当前CND分数（CN环境）
- `get_max_connected_nodes_num()`：获取最大连通分量大小（ND环境）
- `get_weight_efficiency()`：获取权重效率（成本环境）

## 问题求解策略

不同的启发式策略可用于基准测试：

```python
# 随机策略
action = env.random_action()

# 基于中介中心性的策略
action = env.betweenness_action()

# 度中心性策略（自定义实现）
degrees = {node: env.graph.degree(node) for node in env.avail_list}
action = max(degrees.keys(), key=lambda x: degrees[x])
```
