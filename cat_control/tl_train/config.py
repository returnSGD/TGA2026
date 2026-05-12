"""
《猫语心声》 —— RL PPO训练配置

严格遵循 技术策划案v2 §4.1.5 的训练超参数
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# 项目根路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


@dataclass
class TrainConfig:
    """PPO训练配置"""

    # ══ 模型架构（与 rl_clone 一致） ══
    state_dim: int = 422
    personality_dim: int = 8
    intent_num: int = 15
    embed_dim: int = 128
    nhead: int = 4
    ff_dim: int = 256
    num_layers: int = 3
    seq_len: int = 4          # PPO用4帧序列
    dropout: float = 0.1

    # ══ PPO 超参数 ══
    learning_rate: float = 3e-4
    gamma: float = 0.99        # 折扣因子
    gae_lambda: float = 0.95   # GAE λ
    clip_epsilon: float = 0.2  # PPO裁剪范围
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10       # 每轮rollout更新的epoch数
    batch_size: int = 64
    steps_per_rollout: int = 2048
    target_kl: Optional[float] = 0.015  # 早停KL阈值

    # ══ 训练控制 ══
    total_timesteps: int = 2_000_000     # 总环境步数
    save_freq: int = 100_000            # 模型保存间隔（步数）
    eval_freq: int = 50_000             # 评估间隔
    eval_episodes: int = 10             # 每次评估的episode数
    log_freq: int = 1000               # 日志打印间隔

    # ══ 自对弈配置 ══
    num_cats: int = 3           # 自对弈猫咪数量
    cats_share_policy: bool = True  # 多猫共享策略网络

    # ══ BC预训练 ══
    bc_checkpoint: Optional[str] = None  # BC预训练权重路径

    # ══ 设备 ══
    device: str = "cuda"       # cuda / cpu
    seed: int = 42

    # ══ 路径 ══
    model_save_dir: str = os.path.join(BASE_DIR, "rl_train", "checkpoints")
    log_dir: str = os.path.join(BASE_DIR, "rl_train", "logs")
    export_dir: str = os.path.join(BASE_DIR, "rl_train", "export")

    def __post_init__(self):
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)
        # 自动查找BC检查点
        if self.bc_checkpoint is None:
            self.bc_checkpoint = self._find_bc_checkpoint()

    def _find_bc_checkpoint(self) -> Optional[str]:
        """自动查找 rl_clone 训练的最佳BC模型"""
        candidates = [
            os.path.join(BASE_DIR, "rl_clone", "checkpoints", "bc_policy_best.pt"),
            os.path.join(BASE_DIR, "rl_clone", "checkpoints", "bc_policy_final.pt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None


# 猫咪性格配置（从 rl_environment.config 导入）
CAT_PERSONALITIES = {
    "xiaoxue": [0.0, 0.9, 0.1, 0.2, 0.1, 0.0, 0.1, 0.3],  # 怯懦
    "oreo":    [0.8, 0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.7],  # 傲娇
    "orange":  [0.0, 0.0, 0.8, 0.9, 0.7, 0.0, 0.8, 0.1],  # 贪吃+好奇+活泼
}
