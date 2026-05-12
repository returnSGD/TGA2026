"""
《猫语心声》 —— BC 训练数据加载器

从沙盒收集的 .npz 文件加载 (state, action) 对，
划分训练/验证集，构造 PyTorch DataLoader。
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from typing import Tuple, Optional, Dict
from collections import Counter

from .config import (
    STATE_DIM, PERSONALITY_DIM, INTENT_NUM,
    BC_BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT,
    find_latest_bc_data,
)


class CatBehaviorDataset(Dataset):
    """
    (state, action) 数据集，用于行为克隆训练。

    每个样本:
      state: float32[422] — 完整RL状态向量
      personality: float32[8] — 性格嵌入（从 state 前8维提取）
      action: int64 — 意图标签 (0~14)
    """

    def __init__(self, states: np.ndarray, actions: np.ndarray):
        assert states.shape[0] == actions.shape[0], \
            f"样本数不匹配: states {states.shape[0]} vs actions {actions.shape[0]}"
        assert states.shape[1] == STATE_DIM, \
            f"状态维度错误: {states.shape[1]} vs 期望 {STATE_DIM}"

        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.int64)
        self.n_samples = len(self.actions)

        # 提取性格嵌入（前8维）
        self.personalities = self.states[:, :PERSONALITY_DIM].copy()

        # 统计意图分布
        self.intent_counts = Counter(self.actions.tolist())
        self.intent_weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """计算每个样本的权重（用于平衡采样，少数类权重更高）"""
        total = self.n_samples
        weights = np.zeros(self.n_samples, dtype=np.float32)
        for i in range(self.n_samples):
            intent = self.actions[i]
            count = self.intent_counts.get(intent, 1)
            weights[i] = total / (len(self.intent_counts) * count)
        return weights

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.personalities[idx]),
            torch.tensor(self.actions[idx], dtype=torch.long),
        )

    def get_intent_distribution(self) -> Dict[int, Tuple[int, float]]:
        """返回意图分布: {intent_id: (count, percentage)}"""
        return {
            intent: (count, count / self.n_samples * 100)
            for intent, count in sorted(self.intent_counts.items())
        }


def load_bc_data(data_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 .npz 文件加载行为克隆数据。

    返回:
        states: [N, 422] float32
        actions: [N,] int64
        intent_list: [15] str
    """
    if data_path is None:
        data_path = find_latest_bc_data()

    print(f"加载训练数据: {data_path}")
    data = np.load(data_path, allow_pickle=True)

    states = data["states"]
    actions = data["actions"]
    intent_list = data["intent_list"]

    print(f"  样本数: {len(states)}")
    print(f"  状态维度: {states.shape[1]}")
    print(f"  意图数: {len(intent_list)}")

    return states, actions, intent_list


def create_dataloaders(
    states: np.ndarray,
    actions: np.ndarray,
    batch_size: int = BC_BATCH_SIZE,
    use_weighted_sampler: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, CatBehaviorDataset, CatBehaviorDataset]:
    """
    创建训练/验证 DataLoader。

    参数:
        states, actions: 完整数据集
        batch_size: 批次大小
        use_weighted_sampler: 是否使用加权采样（缓解类别不平衡）
        num_workers: DataLoader 工作进程数

    返回:
        train_loader, val_loader, train_dataset, val_dataset
    """
    n = len(states)

    # 随机打乱索引
    indices = np.random.permutation(n)
    split = int(n * TRAIN_SPLIT)

    train_idx = indices[:split]
    val_idx = indices[split:]

    # 创建数据集
    train_dataset = CatBehaviorDataset(states[train_idx], actions[train_idx])
    val_dataset = CatBehaviorDataset(states[val_idx], actions[val_idx])

    # 训练集：可选加权采样
    if use_weighted_sampler:
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(train_dataset.intent_weights).double(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

    # 验证集：不打乱
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  批次大小: {batch_size}")
    print(f"  加权采样: {'是' if use_weighted_sampler else '否'}")

    # 打印意图分布
    print(f"\n  训练集意图分布:")
    for intent_id, (count, pct) in sorted(train_dataset.get_intent_distribution().items()):
        print(f"    {intent_id:2d}: {count:5d} ({pct:5.1f}%)")

    return train_loader, val_loader, train_dataset, val_dataset


def load_and_prepare(
    data_path: str = None,
    batch_size: int = BC_BATCH_SIZE,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, CatBehaviorDataset, CatBehaviorDataset, np.ndarray]:
    """
    一站式：加载数据 + 划分数据集 + 创建 DataLoader。

    返回:
        train_loader, val_loader, train_dataset, val_dataset, intent_list
    """
    states, actions, intent_list = load_bc_data(data_path)

    # 数据质量检查
    assert not np.any(np.isnan(states)), "数据包含 NaN！"
    assert not np.any(np.isinf(states)), "数据包含 Inf！"
    assert actions.min() >= 0 and actions.max() < len(intent_list), \
        f"动作标签越界: [{actions.min()}, {actions.max()}]"

    # 标准化（可选，帮助训练稳定）
    state_mean = states.mean(axis=0, keepdims=True)
    state_std = states.std(axis=0, keepdims=True) + 1e-8
    states_normalized = (states - state_mean) / state_std

    train_loader, val_loader, train_ds, val_ds = create_dataloaders(
        states_normalized, actions, batch_size, use_weighted_sampler
    )

    # 保存标准化参数（推理时需要）
    norm_params = {
        "mean": state_mean,
        "std": state_std,
    }

    return train_loader, val_loader, train_ds, val_ds, intent_list, norm_params
