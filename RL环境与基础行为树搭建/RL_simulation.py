"""
《猫语心声》RL环境与基础行为树 — 快捷入口

用法:
    python RL_simulation.py                  # 默认200 tick模拟
    python RL_simulation.py --ticks 500      # 500 tick模拟
    python RL_simulation.py --visualize      # 含行为树可视化
    python RL_simulation.py --export         # 导出训练数据
    python RL_simulation.py --test-all       # 测试所有意图
    python RL_simulation.py --test-personality  # 性格差异测试
    python RL_simulation.py --debug-bt eat   # 调试指定行为树
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_environment.main import main

if __name__ == "__main__":
    main()
