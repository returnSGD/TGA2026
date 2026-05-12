"""
《猫语心声》 —— RL 策略网络配置与训练超参数
严格遵循 技术策划案v2 §4.1.2 的网络结构定义
"""

import os

# ==================== 模型架构 ====================
STATE_DIM = 422
PERSONALITY_DIM = 8        # 性格嵌入（状态向量前8维）
INTENT_NUM = 15            # 宏观意图数量
EMBED_DIM = 128            # Transformer d_model
NHEAD = 4                  # Multi-head attention heads
FF_DIM = 256               # Feed-forward 维度
NUM_LAYERS = 3             # Transformer Encoder 层数
SEQ_LEN = 1                # 行为克隆用单帧（PPO阶段改为4）
DROPOUT = 0.2

# ==================== 行为克隆训练参数 ====================
BC_EPOCHS = 80             # 训练轮数
BC_BATCH_SIZE = 64
BC_LEARNING_RATE = 3e-4
BC_WEIGHT_DECAY = 1e-5
BC_LR_SCHEDULER_STEP = 20  # 每 N epoch 衰减一次
BC_LR_SCHEDULER_GAMMA = 0.5

# 数据划分
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# Early stopping
EARLY_STOP_PATIENCE = 15   # 验证 loss 不降的容忍 epoch 数
EARLY_STOP_MIN_DELTA = 1e-4

# ==================== PPO 训练参数（阶段B预置） ====================
PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.99           # 折扣因子
PPO_GAE_LAMBDA = 0.95      # GAE λ
PPO_CLIP_EPSILON = 0.2     # PPO clip 范围
PPO_STEPS_PER_ROLLOUT = 2048
PPO_BATCH_SIZE = 64
PPO_EPOCHS_PER_ROLLOUT = 10
PPO_TRAINING_ROUNDS = 500

# ==================== 路径 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "rl_clone", "checkpoints")
ONNX_EXPORT_DIR = os.path.join(BASE_DIR, "rl_clone", "export")

# 自动查找最新 BC 数据
def find_latest_bc_data():
    """在 training_data 目录中查找最新的 bc_data_*.npz 文件"""
    import glob
    pattern = os.path.join(TRAINING_DATA_DIR, "bc_data_*samples_*.npz")
    files = glob.glob(pattern)
    if not files:
        # 回退：匹配任意 bc_data
        pattern = os.path.join(TRAINING_DATA_DIR, "bc_data_*.npz")
        files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"未找到训练数据文件。请先运行沙盒模拟器收集数据：\n"
            f"  python -m rl_environment.main --target-samples 10000 --export"
        )
    # 按修改时间排序，取最新
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

# 创建必要目录
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(ONNX_EXPORT_DIR, exist_ok=True)
