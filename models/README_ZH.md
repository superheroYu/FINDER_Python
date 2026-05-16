[English](README.md) | 中文

# 模型与数据接口（最小实现）

本目录提供 FINDER 策略网络与数据接口的最小实现，围绕 PyTorch + PyTorch Geometric（PyG）。

## 组件

- gnn_arch.py：FinderGNN 及损失（包含一阶重构项）
- policy_net.py：策略网络（DQN 族）
- data_interfaces.py：数据接口最小集
  - networkx_to_pyg_data(graph, aux_features=None) → Data
  - convert_observation_to_pyg_data(obs) → Data
  - create_graph_batch_from_observations(observations) → GraphBatch

GraphBatch 字段：
- node_features: [N, F]
- edge_index: [2, E]
- batch: [N]
- aux_features: [B, 4]
- laplacian: [N, N]
- edge_weight_sum: 标量
- num_graphs: batch 中的图数量
- max_nodes: batch 内单图最大节点数
- node_ids: [N]

## 观测约定

训练器/环境通过字典传递观测：
- 必需：'graph'（NetworkX 图）
- 可选：'aux_features'（np.ndarray(4,)；标准为 [covered_ratio, edge_covered_ratio, twohop_density, 1.0]）
其余键不再使用（如 available_actions / action_mask / covered_nodes / current_step）。

## 批处理

使用 create_graph_batch_from_observations 将观测列表直接拼接为 GraphBatch。
- 若节点特征缺失，批内用零张量兜底为 [N, 2]，保证模型前向可执行。
- laplacian 按子图块对角拼接。

## 与训练器对接

训练器从 envs.make_batch_env（内部：gymnasium.vector）获取 List[Dict] 观测，
经 data_interfaces 转换为 GraphBatch，输入策略网络进行动作选择与训练。

## 版本

- 最小实现；无动作掩码与可用动作列表；无环境兼容检查函数。


