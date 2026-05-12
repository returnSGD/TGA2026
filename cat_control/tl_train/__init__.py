"""
《猫语心声》 —— RL策略网络训练模块

阶段二（第5-8周）：PPO训练框架
- 单猫PPO训练
- 多猫自对弈训练
- 性格条件化策略（FiLM调制）
- BC预训练权重加载
- ONNX导出
"""

from .config import TrainConfig
from .ppo import PPO
from .env_wrapper import MultiCatEnv, SingleCatEnv
from .trainer import PPOTrainer
