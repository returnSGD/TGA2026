"""
rl_clone — 《猫语心声》行为克隆预训练模块

阶段A：使用规则策略收集的 (state, action) 数据，以交叉熵损失预训练 RL 策略网络。
"""

from .config import (
    STATE_DIM, PERSONALITY_DIM, INTENT_NUM, EMBED_DIM,
    NHEAD, FF_DIM, NUM_LAYERS, SEQ_LEN, DROPOUT,
    BC_EPOCHS, BC_BATCH_SIZE, BC_LEARNING_RATE,
    MODEL_SAVE_DIR, find_latest_bc_data,
)
from .model import RLPolicyNetwork, FiLMModulation
from .data_loader import CatBehaviorDataset, load_and_prepare
from .train_bc import train_bc, save_checkpoint, validate, export_onnx
