"""
《猫语心声》 —— RL 策略网络模型

严格遵循 技术策划案v2 §4.1.2 定义：
  输入投影 → Transformer Encoder → FiLM调制 → Actor/Critic Head

结构:
  state_seq [B, S, 422] → Linear+LayerNorm → [B, S, 128]
  + learnable pos_embed [1, S, 128]
  → TransformerEncoder (3层, d=128, nhead=4, ff=256)
  → 取最后时间步 [B, 128]
  → FiLM: γ,β = PersonalityMLP(personality_embed[8]) → h_modulated
  → Actor Head: Linear(128→64) + ReLU + Linear(64→15)
  → Critic Head: Linear(128→64) + ReLU + Linear(64→1)

总参数量 ≈ 0.8M，CPU 推理 <2ms
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FiLMModulation(nn.Module):
    """
    性格嵌入的 FiLM (Feature-wise Linear Modulation) 条件化模块。

    γ = W_γ · personality_embed
    β = W_β · personality_embed
    output = γ ⊙ input + β

    使同一策略网络能依据性格向量输出不同的行为分布。
    """

    def __init__(self, embed_dim: int = 128, personality_dim: int = 8):
        super().__init__()
        self.gamma_net = nn.Linear(personality_dim, embed_dim)
        self.beta_net = nn.Linear(personality_dim, embed_dim)

        # 初始化：γ 接近 1，β 接近 0，避免初始训练时大幅扰动
        nn.init.constant_(self.gamma_net.weight, 0.0)
        nn.init.constant_(self.gamma_net.bias, 1.0)
        nn.init.constant_(self.beta_net.weight, 0.0)
        nn.init.constant_(self.beta_net.bias, 0.0)

    def forward(self, x: torch.Tensor,
                personality_embed: torch.Tensor) -> torch.Tensor:
        """
        x: [B, embed_dim]  Transformer 编码后的特征
        personality_embed: [B, personality_dim]  性格向量
        返回: [B, embed_dim] 调制后的特征
        """
        gamma = self.gamma_net(personality_embed)
        beta = self.beta_net(personality_embed)
        return gamma * x + beta


class RLPolicyNetwork(nn.Module):
    """
    PPO 策略网络：Transformer Encoder + FiLM + Actor/Critic Head。

    输入:  state_seq [B, S, 422] — 当前 + 前 S-1 步状态
           personality_embed [B, 8] — 性格嵌入向量（条件化网络行为）
    输出:  action_logits [B, 15] — 各意图的 logits
           state_value [B, 1] — 状态价值 V(s)
    """

    def __init__(self,
                 state_dim: int = 422,
                 embed_dim: int = 128,
                 num_intents: int = 15,
                 seq_len: int = 1,          # BC 阶段用 1，PPO 阶段用 4
                 personality_dim: int = 8,
                 nhead: int = 4,
                 ff_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.num_intents = num_intents
        self.seq_len = seq_len
        self.personality_dim = personality_dim

        # ══ 输入投影 ══
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ══ 可学习位置编码 ══
        self.pos_embed = nn.Parameter(
            torch.randn(1, seq_len, embed_dim) * 0.02
        )

        # ══ Transformer Encoder ══
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ══ FiLM 性格调制 ══
        self.film = FiLMModulation(embed_dim, personality_dim)

        # ══ Actor Head: 输出各意图的动作概率 ══
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_intents),
        )

        # ══ Critic Head: 输出状态价值 V(s) ══
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot 初始化，actor/critic 最后层小权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Actor 和 Critic 最后一层用小权重，使初始策略接近均匀分布
        last_actor = self.actor_head[-1]
        if isinstance(last_actor, nn.Linear):
            nn.init.xavier_uniform_(last_actor.weight, gain=0.01)
        last_critic = self.critic_head[-1]
        if isinstance(last_critic, nn.Linear):
            nn.init.xavier_uniform_(last_critic.weight, gain=0.01)

    def forward(self, state_seq: torch.Tensor,
                personality_embed: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        参数:
            state_seq: [B, S, state_dim]  状态序列
            personality_embed: [B, personality_dim]  性格嵌入

        返回:
            action_logits: [B, num_intents]
            state_value: [B, 1]
        """
        B = state_seq.size(0)

        # 1. 输入投影
        x = self.state_proj(state_seq)              # [B, S, E]

        # 2. 位置编码
        pos = self.pos_embed[:, :x.size(1), :]      # [1, S, E]
        x = x + pos

        # 3. Transformer Encoder
        x = self.encoder(x)                         # [B, S, E]

        # 4. 取最后一个时间步
        x_last = x[:, -1, :]                        # [B, E]

        # 5. FiLM 性格调制
        x_modulated = self.film(x_last, personality_embed)  # [B, E]

        # 6. Actor / Critic
        action_logits = self.actor_head(x_modulated)        # [B, 15]
        state_value = self.critic_head(x_modulated).squeeze(-1)  # [B]

        return action_logits, state_value

    def forward_single_state(self, state: torch.Tensor,
                             personality_embed: torch.Tensor = None
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单状态前向传播（简化接口，用于推理和 BC 训练）。

        state: [B, state_dim]
        personality_embed: [B, personality_dim] — 若为 None 则从 state 前8维提取
        """
        if personality_embed is None:
            personality_embed = state[:, :self.personality_dim]

        # 添加序列维度
        state_seq = state.unsqueeze(1)  # [B, 1, state_dim]
        return self.forward(state_seq, personality_embed)

    def get_action(self, state: torch.Tensor,
                   personality_embed: torch.Tensor = None,
                   deterministic: bool = False
                   ) -> Tuple[int, torch.Tensor]:
        """
        推理时获取动作（离散采样或 argmax）。

        返回:
            action_idx: int
            probs: [15] 动作概率分布
        """
        with torch.no_grad():
            logits, value = self.forward_single_state(
                state.unsqueeze(0),  # [1, state_dim]
                personality_embed.unsqueeze(0) if personality_embed is not None else None,
            )
            probs = F.softmax(logits, dim=-1).squeeze(0)

            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = torch.multinomial(probs, 1).item()

        return action, probs

    def count_parameters(self) -> int:
        """统计总参数量"""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """统计可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """模型结构摘要"""
        total = self.count_parameters()
        trainable = self.count_trainable_parameters()
        lines = [
            f"RLPolicyNetwork (总参数: {total:,}, 可训练: {trainable:,})",
            f"  输入: state[{self.state_dim}] + personality[{self.personality_dim}]",
            f"  序列长度: {self.seq_len}",
            f"  Embed dim: {self.embed_dim}, Heads: {4}, FF: {256}",
            f"  Transformer 层数: {3}",
            f"  输出: action_logits[{self.num_intents}] + value[1]",
        ]
        return "\n".join(lines)
