"""
《猫语心声》 —— 行为克隆（BC）训练

阶段A：使用规则策略收集的 (state, action) 数据，
      以交叉熵损失预训练 RL 策略网络，使其初始行为接近规则策略。

训练流程:
  1. 加载 .npz 数据 → 标准化 → 划分 train/val
  2. 以交叉熵损失训练 Actor Head（Critic 不参与 BC 阶段）
  3. 每 epoch 验证，记录最佳模型
  4. 保存 checkpoint + 导出 ONNX 格式
"""

from __future__ import annotations
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict, Optional

from .config import (
    STATE_DIM, PERSONALITY_DIM, INTENT_NUM, EMBED_DIM,
    NHEAD, FF_DIM, NUM_LAYERS, SEQ_LEN, DROPOUT,
    BC_EPOCHS, BC_LEARNING_RATE, BC_WEIGHT_DECAY,
    BC_LR_SCHEDULER_STEP, BC_LR_SCHEDULER_GAMMA,
    EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA,
    MODEL_SAVE_DIR, ONNX_EXPORT_DIR,
)
from .model import RLPolicyNetwork
from .data_loader import load_and_prepare, CatBehaviorDataset


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """计算 Top-1 准确率"""
    preds = torch.argmax(logits, dim=-1)
    return (preds == targets).float().mean().item()


def compute_per_class_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                               num_classes: int = INTENT_NUM) -> Dict[int, Tuple[int, int, float]]:
    """计算每个意图类别的准确率"""
    preds = torch.argmax(logits, dim=-1)
    results = {}
    for c in range(num_classes):
        mask = (targets == c)
        total = mask.sum().item()
        if total > 0:
            correct = (preds[mask] == c).sum().item()
            results[c] = (correct, total, correct / total * 100)
        else:
            results[c] = (0, 0, 0.0)
    return results


def train_one_epoch(model: RLPolicyNetwork,
                    dataloader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    epoch: int,
                    total_epochs: int) -> Dict:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = len(dataloader)

    for batch_idx, (states, personalities, actions) in enumerate(dataloader):
        states = states.to(device)
        personalities = personalities.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()

        # 前向传播（单帧模式）
        logits, _ = model.forward_single_state(states, personalities)
        loss = criterion(logits, actions)

        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += compute_accuracy(logits, actions)

        # 进度条
        if batch_idx % 20 == 0 or batch_idx == n_batches - 1:
            print(f"\r  Epoch {epoch:3d}/{total_epochs} "
                  f"[{batch_idx:4d}/{n_batches}] "
                  f"loss: {loss.item():.4f} "
                  f"acc: {total_acc / (batch_idx + 1):.3f}",
                  end="", flush=True)

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    return {"loss": avg_loss, "accuracy": avg_acc}


@torch.no_grad()
def validate(model: RLPolicyNetwork,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Dict:
    """验证"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = len(dataloader)
    all_preds = []
    all_targets = []

    for states, personalities, actions in dataloader:
        states = states.to(device)
        personalities = personalities.to(device)
        actions = actions.to(device)

        logits, _ = model.forward_single_state(states, personalities)
        loss = criterion(logits, actions)

        total_loss += loss.item()
        total_acc += compute_accuracy(logits, actions)
        all_preds.append(torch.argmax(logits, dim=-1).cpu())
        all_targets.append(actions.cpu())

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    # 逐类准确率
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    per_class = compute_per_class_accuracy(
        torch.nn.functional.one_hot(all_preds, INTENT_NUM).float(),
        all_targets,
    )

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "per_class_accuracy": per_class,
    }


def save_checkpoint(model: RLPolicyNetwork, optimizer: optim.Optimizer,
                    epoch: int, train_metrics: Dict, val_metrics: Dict,
                    filepath: str):
    """保存训练检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "model_config": {
            "state_dim": model.state_dim,
            "embed_dim": model.embed_dim,
            "num_intents": model.num_intents,
            "seq_len": model.seq_len,
            "personality_dim": model.personality_dim,
        },
    }
    torch.save(checkpoint, filepath)


def export_onnx(model: RLPolicyNetwork, filepath: str,
                device: torch.device):
    """导出 ONNX 格式（用于 Unity Barracuda 或 C++ 推理）"""
    model.eval()
    dummy_state = torch.randn(1, 1, STATE_DIM, device=device)
    dummy_personality = torch.randn(1, PERSONALITY_DIM, device=device)

    torch.onnx.export(
        model,
        (dummy_state, dummy_personality),
        filepath,
        input_names=["state_seq", "personality_embed"],
        output_names=["action_logits", "state_value"],
        dynamic_axes={
            "state_seq": {0: "batch"},
            "personality_embed": {0: "batch"},
            "action_logits": {0: "batch"},
            "state_value": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"\n  ONNX 模型已导出: {filepath}")


# ==================== 主训练函数 ====================

def train_bc(
    data_path: str = None,
    epochs: int = BC_EPOCHS,
    batch_size: int = None,
    learning_rate: float = BC_LEARNING_RATE,
    device: torch.device = None,
    use_weighted_sampler: bool = True,
    save_prefix: str = "bc_policy",
    export_onnx_model: bool = True,
) -> Tuple[RLPolicyNetwork, Dict]:
    """
    执行行为克隆训练。

    参数:
        data_path: .npz 数据路径（默认自动查找最新）
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        device: 计算设备
        use_weighted_sampler: 是否使用加权采样
        save_prefix: 模型保存文件名前缀
        export_onnx_model: 是否导出 ONNX

    返回:
        (训练好的模型, 训练历史记录)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        from .config import BC_BATCH_SIZE
        batch_size = BC_BATCH_SIZE

    print(f"\n{'═' * 60}")
    print(f"  《猫语心声》行为克隆（BC）训练")
    print(f"  阶段A：RL策略网络预热")
    print(f"{'═' * 60}")
    print(f"  设备: {device}")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  目标轮数: {epochs}")

    # ── 加载数据 ──
    train_loader, val_loader, train_ds, val_ds, intent_list, norm_params = \
        load_and_prepare(data_path, batch_size, use_weighted_sampler)

    # ── 创建模型 ──
    model = RLPolicyNetwork(
        state_dim=STATE_DIM,
        embed_dim=EMBED_DIM,
        num_intents=INTENT_NUM,
        seq_len=SEQ_LEN,
        personality_dim=PERSONALITY_DIM,
        nhead=NHEAD,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    print(f"\n{model.summary()}")
    print(f"  实际参数量: {model.count_parameters():,}")

    # ── 损失函数 & 优化器 ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                           weight_decay=BC_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                         step_size=BC_LR_SCHEDULER_STEP,
                                         gamma=BC_LR_SCHEDULER_GAMMA)

    # ── 训练循环 ──
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'─' * 60}")
    print(f"  开始训练")
    print(f"{'─' * 60}")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, epochs
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])

        # 学习率调度
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        # 打印
        print(f"\r  Epoch {epoch:3d}/{epochs} | "
              f"train loss: {train_metrics['loss']:.4f} acc: {train_metrics['accuracy']:.3f} | "
              f"val loss: {val_metrics['loss']:.4f} acc: {val_metrics['accuracy']:.3f} | "
              f"lr: {current_lr:.2e}")

        # ── 保存最佳模型 ──
        if val_metrics["loss"] < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

            best_path = os.path.join(MODEL_SAVE_DIR, f"{save_prefix}_best.pt")
            save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics, best_path)
            print(f"    → 保存最佳模型 (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1

        # ── Early Stopping ──
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping: val_loss 连续 {EARLY_STOP_PATIENCE} 轮未改善")
            break

    elapsed = time.time() - start_time

    # ── 训练完成 ──
    print(f"\n{'─' * 60}")
    print(f"  训练完成 ({elapsed:.0f}s)")
    print(f"  最佳val_loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"  最终train_acc: {history['train_acc'][-1]:.3f}")
    print(f"  最终val_acc: {history['val_acc'][-1]:.3f}")
    print(f"{'─' * 60}")

    # ── 加载最佳模型 ──
    best_path = os.path.join(MODEL_SAVE_DIR, f"{save_prefix}_best.pt")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  已加载最佳模型: {best_path}")

    # ── 最终模型保存 ──
    final_path = os.path.join(MODEL_SAVE_DIR, f"{save_prefix}_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "state_dim": STATE_DIM,
            "embed_dim": EMBED_DIM,
            "num_intents": INTENT_NUM,
            "seq_len": SEQ_LEN,
            "personality_dim": PERSONALITY_DIM,
        },
        "norm_params": norm_params,
        "intent_list": intent_list.tolist() if isinstance(intent_list, np.ndarray) else list(intent_list),
        "training_history": history,
    }, final_path)
    print(f"  最终模型已保存: {final_path}")

    # ── 保存训练历史 JSON ──
    history_path = os.path.join(MODEL_SAVE_DIR, f"{save_prefix}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  训练历史已保存: {history_path}")

    # ── 导出 ONNX ──
    if export_onnx_model:
        onnx_path = os.path.join(ONNX_EXPORT_DIR, f"{save_prefix}.onnx")
        try:
            export_onnx(model, onnx_path, device)
        except Exception as e:
            print(f"\n  ONNX 导出失败（模型已保存，可后续手动导出）: {e}")

    # ── 逐类准确率报告 ──
    print(f"\n{'─' * 60}")
    print(f"  逐类验证准确率")
    print(f"{'─' * 60}")
    val_metrics = validate(model, val_loader, criterion, device)
    per_class = val_metrics["per_class_accuracy"]
    intent_names = intent_list if isinstance(intent_list, list) else intent_list.tolist()

    for c in sorted(per_class.keys()):
        correct, total, acc = per_class[c]
        intent_name = intent_names[c] if c < len(intent_names) else f"intent_{c}"
        bar = "█" * int(acc / 100 * 20)
        print(f"  {intent_name:22s}: {correct:4d}/{total:4d} ({acc:5.1f}%) {bar}")

    print(f"\n{'═' * 60}")
    print(f"  行为克隆训练完成！")
    print(f"  下一步：使用 BC 预训练权重初始化 PPO 策略网络（阶段B）")
    print(f"{'═' * 60}\n")

    return model, history
