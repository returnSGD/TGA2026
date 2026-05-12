"""
《猫语心声》 —— BC 训练入口

用法:
    python -m rl_clone.main [--data-path PATH] [--epochs N] [--batch-size N] [--lr LR] [--device DEVICE]

示例:
    python -m rl_clone.main --epochs 80 --batch-size 64
    python -m rl_clone.main --data-path training_data/bc_data_10149samples_xxx.npz --epochs 30
    python -m rl_clone.main --device cpu --epochs 10  # 快速测试
"""

from __future__ import annotations
import argparse
import sys
import torch

from .config import (
    BC_EPOCHS, BC_BATCH_SIZE, BC_LEARNING_RATE,
    find_latest_bc_data,
)
from .train_bc import train_bc


def parse_args():
    parser = argparse.ArgumentParser(
        description="《猫语心声》行为克隆（BC）预训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m rl_clone.main --epochs 80 --batch-size 64
  python -m rl_clone.main --data-path training_data/bc_data_xxx.npz --epochs 30
  python -m rl_clone.main --device cpu --epochs 10 --no-export
        """,
    )

    parser.add_argument(
        "--data-path", type=str, default=None,
        help="训练数据 .npz 路径（默认自动查找最新）",
    )
    parser.add_argument(
        "--epochs", type=int, default=BC_EPOCHS,
        help=f"训练轮数（默认: {BC_EPOCHS}）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BC_BATCH_SIZE,
        help=f"批次大小（默认: {BC_BATCH_SIZE}）",
    )
    parser.add_argument(
        "--lr", type=float, default=BC_LEARNING_RATE,
        help=f"学习率（默认: {BC_LEARNING_RATE}）",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="计算设备: cuda / cpu（默认自动检测）",
    )
    parser.add_argument(
        "--no-weighted", action="store_true",
        help="禁用加权采样（默认开启）",
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="不导出 ONNX 模型",
    )
    parser.add_argument(
        "--save-prefix", type=str, default="bc_policy",
        help="模型保存文件名前缀（默认: bc_policy）",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = None
    if args.device:
        device = torch.device(args.device)

    print(f"\n  BC 训练入口")
    print(f"  数据路径: {args.data_path or '自动查找最新'}")
    print(f"  设备: {args.device or '自动检测'}")
    print(f"  轮数: {args.epochs}")
    print(f"  批次: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  加权采样: {'否' if args.no_weighted else '是'}")
    print(f"  ONNX导出: {'否' if args.no_export else '是'}")

    try:
        model, history = train_bc(
            data_path=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            use_weighted_sampler=not args.no_weighted,
            save_prefix=args.save_prefix,
            export_onnx_model=not args.no_export,
        )
    except FileNotFoundError as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n训练被用户中断。")
        sys.exit(0)

    print(f"\n最终验证准确率: {history['val_acc'][-1]:.3f}")
    print(f"模型已保存至 rl_clone/checkpoints/ 和 rl_clone/export/")


if __name__ == "__main__":
    main()
