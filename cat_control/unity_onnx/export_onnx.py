"""
《猫语心声》 —— 批量 .pt → ONNX 模型导出脚本

将所有已训练的 RL 策略网络检查点转换为 ONNX 格式，
供 Unity Barracuda / C++ 推理使用。

用法:
    python export_onnx.py                    # 导出所有 .pt 文件
    python export_onnx.py --checkpoint PATH  # 导出指定检查点
    python export_onnx.py --verify           # 验证已导出的 ONNX 模型
"""

from __future__ import annotations
import sys
import os
import argparse
import glob
import numpy as np
import torch
from typing import Dict, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from rl_clone.model import RLPolicyNetwork
from rl_environment.config import INTENT_LIST, PERSONALITY_DIM, STATE_DIM

# ─── 路径配置 ───
BC_CHECKPOINTS_DIR = os.path.join(BASE_DIR, "rl_clone", "checkpoints")
BC_EXPORT_DIR = os.path.join(BASE_DIR, "rl_clone", "export")
PPO_CHECKPOINTS_DIR = os.path.join(BASE_DIR, "rl_train", "checkpoints")
PPO_EXPORT_DIR = os.path.join(BASE_DIR, "rl_train", "export")
UNITY_EXPORT_DIR = os.path.join(BASE_DIR, "unity_onnx")  # Unity 统一导出目录


def detect_checkpoint_type(checkpoint: Dict) -> Tuple[str, int]:
    """
    自动检测检查点类型。

    返回: ("bc"|"ppo", seq_len)
    """
    # BC 检查点有 "model_config" 键
    if "model_config" in checkpoint:
        mc = checkpoint["model_config"]
        seq_len = mc.get("seq_len", 1)
        return "bc", seq_len

    # PPO 检查点有 "config" 键
    if "config" in checkpoint:
        cfg = checkpoint["config"]
        seq_len = cfg.get("seq_len", 4)
        return "ppo", seq_len

    # 回退：通过键推断
    if "epoch" in checkpoint:
        return "bc", 1
    if "total_steps" in checkpoint:
        return "ppo", 4

    # 最终回退
    return "unknown", 1


def create_model_from_checkpoint(checkpoint: Dict) -> RLPolicyNetwork:
    """从检查点创建匹配的 RLPolicyNetwork 模型"""
    ckpt_type, seq_len = detect_checkpoint_type(checkpoint)

    # 提取配置
    if "model_config" in checkpoint:
        mc = checkpoint["model_config"]
    elif "config" in checkpoint:
        mc = checkpoint["config"]
    else:
        mc = {}

    state_dim = mc.get("state_dim", STATE_DIM)
    embed_dim = mc.get("embed_dim", 128)
    num_intents = mc.get("num_intents", 15)
    personality_dim = mc.get("personality_dim", PERSONALITY_DIM)
    nhead = mc.get("nhead", 4) if "nhead" in mc else 4
    ff_dim = mc.get("ff_dim", 256) if "ff_dim" in mc else 256
    num_layers = mc.get("num_layers", 3) if "num_layers" in mc else 3
    dropout = mc.get("dropout", 0.1) if "dropout" in mc else 0.1

    print(f"  模型配置: type={ckpt_type}, seq_len={seq_len}, state_dim={state_dim}")
    print(f"  架构: d={embed_dim}, nhead={nhead}, ff={ff_dim}, layers={num_layers}")

    model = RLPolicyNetwork(
        state_dim=state_dim,
        embed_dim=embed_dim,
        num_intents=num_intents,
        seq_len=seq_len,
        personality_dim=personality_dim,
        nhead=nhead,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    return model


def load_checkpoint(checkpoint_path: str) -> Tuple[RLPolicyNetwork, Dict, str, int]:
    """加载检查点并返回 (模型, 检查点字典, 类型, seq_len)"""
    print(f"  加载: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_type, seq_len = detect_checkpoint_type(checkpoint)

    model = create_model_from_checkpoint(checkpoint)

    # 加载权重
    model_state = checkpoint.get("model_state_dict", checkpoint)
    if "model_state_dict" not in checkpoint:
        # 可能整个 checkpoint 就是 state_dict
        pass

    # 过滤不匹配的键
    policy_dict = model.state_dict()
    filtered_state = {}
    skipped = []
    for k, v in model_state.items():
        if k in policy_dict:
            if v.shape == policy_dict[k].shape:
                filtered_state[k] = v
            else:
                skipped.append(f"{k}: {v.shape} vs {policy_dict[k].shape}")
        else:
            skipped.append(k)

    if skipped:
        print(f"  跳过 {len(skipped)} 个不匹配的键")
        if len(skipped) <= 5:
            for s in skipped:
                print(f"    - {s}")

    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  加载成功: {len(filtered_state)}/{len(policy_dict)} 参数匹配, "
          f"总参数: {n_params:,}")

    return model, checkpoint, ckpt_type, seq_len


def export_to_onnx(model: RLPolicyNetwork,
                   output_path: str,
                   seq_len: int,
                   state_dim: int = STATE_DIM,
                   personality_dim: int = PERSONALITY_DIM,
                   opset_version: int = 14) -> bool:
    """
    导出模型为 ONNX 格式。

    ONNX 输入:
      - state_seq: [batch, seq_len, state_dim]  状态序列
      - personality_embed: [batch, personality_dim]  性格嵌入

    ONNX 输出:
      - action_logits: [batch, 15]  各意图的 logits
      - state_value: [batch]  状态价值 V(s)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model.eval()

    # 构造虚拟输入
    dummy_state = torch.randn(1, seq_len, state_dim)
    dummy_personality = torch.randn(1, personality_dim)

    try:
        torch.onnx.export(
            model,
            (dummy_state, dummy_personality),
            output_path,
            input_names=["state_seq", "personality_embed"],
            output_names=["action_logits", "state_value"],
            dynamic_axes={
                "state_seq": {0: "batch"},
                "personality_embed": {0: "batch"},
                "action_logits": {0: "batch"},
                "state_value": {0: "batch"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
        )
        print(f"  [OK] 导出成功: {output_path}")
        return True
    except Exception as e:
        print(f"  [FAIL] 导出失败: {e}")
        return False


def verify_onnx(onnx_path: str) -> bool:
    """验证 ONNX 模型"""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        # 打印信息
        input_names = [inp.name for inp in model.graph.input]
        output_names = [out.name for out in model.graph.output]
        print(f"  [OK] 验证通过: {os.path.basename(onnx_path)}")
        print(f"    输入: {input_names}")
        print(f"    输出: {output_names}")
        print(f"    大小: {os.path.getsize(onnx_path) / (1024*1024):.1f} MB")
        return True
    except ImportError:
        print(f"  ⚠ onnx 包未安装，跳过验证")
        return True
    except Exception as e:
        print(f"  [FAIL] 验证失败: {e}")
        return False


def export_all_checkpoints(export_to_unity: bool = True):
    """导出所有 .pt 检查点为 ONNX"""
    print(f"\n{'═' * 60}")
    print(f"  《猫语心声》批量 .pt → ONNX 导出")
    print(f"{'═' * 60}")

    # 收集所有 .pt 文件
    all_pt_files = []
    for search_dir in [BC_CHECKPOINTS_DIR, PPO_CHECKPOINTS_DIR]:
        if os.path.isdir(search_dir):
            pt_files = glob.glob(os.path.join(search_dir, "*.pt"))
            all_pt_files.extend(pt_files)

    if not all_pt_files:
        print("未找到任何 .pt 文件！")
        return

    print(f"\n找到 {len(all_pt_files)} 个 .pt 检查点")

    results = {"success": [], "failed": []}

    for pt_path in sorted(all_pt_files):
        basename = os.path.splitext(os.path.basename(pt_path))[0]

        # 判断所属目录
        if "rl_clone" in pt_path.replace("\\", "/"):
            export_dir = BC_EXPORT_DIR
            prefix = "bc"
        else:
            export_dir = PPO_EXPORT_DIR
            prefix = "ppo"

        onnx_filename = f"{basename}.onnx"
        onnx_path = os.path.join(export_dir, onnx_filename)

        print(f"\n{'─' * 60}")
        print(f"  [{basename}]")

        try:
            model, checkpoint, ckpt_type, seq_len = load_checkpoint(pt_path)

            # 导出到模块目录
            success = export_to_onnx(model, onnx_path, seq_len)

            if success:
                # 也导出到 Unity 统一目录
                if export_to_unity:
                    os.makedirs(UNITY_EXPORT_DIR, exist_ok=True)
                    unity_path = os.path.join(UNITY_EXPORT_DIR, onnx_filename)
                    export_to_onnx(model, unity_path, seq_len)

                results["success"].append(onnx_path)

                # 验证
                if os.path.exists(onnx_path):
                    verify_onnx(onnx_path)
            else:
                results["failed"].append(pt_path)

        except Exception as e:
            print(f"  [FAIL] 处理失败: {e}")
            import traceback
            traceback.print_exc()
            results["failed"].append(pt_path)

    # ── 汇总 ──
    print(f"\n{'═' * 60}")
    print(f"  导出完成")
    print(f"  成功: {len(results['success'])} / {len(all_pt_files)}")
    print(f"{'═' * 60}")

    if results["success"]:
        print(f"\n  已导出的 ONNX 模型:")
        for p in results["success"]:
            size_mb = os.path.getsize(p) / (1024 * 1024) if os.path.exists(p) else 0
            print(f"    {p} ({size_mb:.1f} MB)")

    if results["failed"]:
        print(f"\n  导出失败:")
        for p in results["failed"]:
            print(f"    {p}")

    # ── Unity 集成说明 ──
    if export_to_unity:
        print(f"\n{'─' * 60}")
        print(f"  Unity 集成指引")
        print(f"{'─' * 60}")
        unity_guide = f"""
  ONNX 模型已导出到: {UNITY_EXPORT_DIR}

  在 Unity 中使用 Barracuda 加载:

  1. 安装 Unity Barracuda 包:
     Package Manager -> Add package by name -> com.unity.barracuda

  2. 加载模型:
     using Unity.Barracuda;
     IWorker m_Worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpBurst, model);

  3. 推理输入:
     - state_seq [1, seq_len, 422]: 状态序列
     - personality_embed [1, 8]: 性格嵌入

  4. 推荐模型选择:
     - Unity 运行时推理: 使用 BC 模型 (bc_policy_best.onnx), seq_len=1, 推理最快
     - 高质量离线推理: 使用 PPO 模型 (ppo_single_oreo_best.onnx), seq_len=4

  5. 性格嵌入配置:
     - 小雪 (xiaoxue, 怯懦): [0.0, 0.9, 0.1, 0.2, 0.1, 0.0, 0.1, 0.3]
     - 奥利奥 (oreo, 傲娇): [0.8, 0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.7]
     - 橘子 (orange, 贪吃): [0.0, 0.0, 0.8, 0.9, 0.7, 0.0, 0.8, 0.1]
  """
        print(unity_guide)


def export_single_checkpoint(checkpoint_path: str, output_path: str = None):
    """导出单个检查点"""
    if output_path is None:
        basename = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_path = os.path.join(UNITY_EXPORT_DIR, f"{basename}.onnx")

    print(f"\n{'═' * 60}")
    print(f"  导出单个检查点")
    print(f"{'═' * 60}")

    model, checkpoint, ckpt_type, seq_len = load_checkpoint(checkpoint_path)

    # 额外信息
    if ckpt_type == "bc":
        epoch = checkpoint.get("epoch", "?")
        print(f"  BC 检查点 | Epoch: {epoch}")
    elif ckpt_type == "ppo":
        steps = checkpoint.get("total_steps", "?")
        updates = checkpoint.get("total_updates", "?")
        best_reward = checkpoint.get("best_reward", "?")
        print(f"  PPO 检查点 | Steps: {steps} | Updates: {updates} | Best Reward: {best_reward}")

    success = export_to_onnx(model, output_path, seq_len)
    if success:
        verify_onnx(output_path)


def verify_all_onnx():
    """验证所有已导出的 ONNX 模型"""
    print(f"\n{'═' * 60}")
    print(f"  验证所有 ONNX 模型")
    print(f"{'═' * 60}")

    all_onnx = []
    for d in [BC_EXPORT_DIR, PPO_EXPORT_DIR, UNITY_EXPORT_DIR]:
        if os.path.isdir(d):
            all_onnx.extend(glob.glob(os.path.join(d, "*.onnx")))

    if not all_onnx:
        print("未找到 ONNX 模型！")
        return

    for onnx_path in sorted(all_onnx):
        print(f"\n  {onnx_path}")
        verify_onnx(onnx_path)


# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="《猫语心声》RL策略网络 .pt → ONNX 批量导出工具"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="导出单个检查点（不指定则批量导出全部）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="ONNX 输出路径（仅 --checkpoint 模式）"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="验证已导出的 ONNX 模型"
    )
    parser.add_argument(
        "--no-unity", action="store_true",
        help="不导出到 Unity 统一目录"
    )
    parser.add_argument(
        "--opset", type=int, default=14,
        help="ONNX opset 版本（默认: 14）"
    )

    args = parser.parse_args()

    if args.verify:
        verify_all_onnx()
    elif args.checkpoint:
        export_single_checkpoint(args.checkpoint, args.output)
    else:
        export_all_checkpoints(export_to_unity=not args.no_unity)


if __name__ == "__main__":
    main()