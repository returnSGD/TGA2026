# ================================================================
# 《猫语心声》核心系统伪代码
# 对应架构：猫咪个体控制流 + LLM叙事控制流
# ================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import heapq


# ================================================================
# 一、基础数据结构
# ================================================================

class ActionIntent(Enum):
    """猫咪可执行的动作意图枚举"""
    RUB_LEG    = "蹭腿"
    MEOW       = "叫唤"
    HIDE       = "躲藏"
    PURR       = "呼噜"
    PLAY       = "玩耍"
    EAT        = "进食"
    SLEEP      = "睡觉"
    STARE      = "凝视"
    FLEE       = "逃跑"


class PlayerAction(Enum):
    """玩家可触发的交互动作"""
    PET        = "抚摸"
    FEED       = "喂食"
    PLAY       = "玩耍"
    TALK       = "说话"
    # 关键叙事节点触发
    STORY_NODE = "主线推进"
    RESCUE     = "救助流浪猫"


@dataclass
class EmotionTag:
    """情感/话题标签，用于驱动轻量LLM生成猫语"""
    emotion: str        # e.g. "开心", "戒备", "依赖", "委屈"
    topic: str          # e.g. "食物", "童年创伤", "主人", "陌生人"
    intensity: float    # 0.0 ~ 1.0


@dataclass
class MemoryEntry:
    """单条记忆条目"""
    event_description: str
    emotional_weight: float     # 情感权重，决定优先级队列排序
    timestamp: int              # 游戏内时间戳
    is_pinned: bool = False     # 核心记忆是否永久锚定

    # 用于优先级队列比较（权重越高越优先保留）
    def __lt__(self, other: MemoryEntry):
        return self.emotional_weight > other.emotional_weight


@dataclass
class PersonalityParams:
    """猫咪性格参数（由设计师配置，全生命周期稳定）"""
    name: str
    clingy: float       # 黏人度      0~1
    timid: float        # 胆怯度      0~1
    greedy: float       # 贪吃度      0~1
    playful: float      # 活泼度      0~1
    trauma_tags: list[str] = field(default_factory=list)   # 创伤标签
    backstory: str = ""                                     # 背景故事文本


# ================================================================
# 二、记忆模块（固定窗口 + 优先级队列）
# ================================================================

class MemoryModule:
    """
    短期记忆：固定大小滑动窗口（近期事件）
    长期记忆：优先级队列（高情感权重事件永久留存）
    """
    def __init__(self, short_term_capacity=20, long_term_capacity=100):
        self.short_term: list[MemoryEntry] = []            # 滑动窗口
        self.long_term: list[MemoryEntry] = []             # 最大堆
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity  = long_term_capacity

    def add_memory(self, entry: MemoryEntry):
        # 短期记忆：超出窗口则移除最旧的
        self.short_term.append(entry)
        if len(self.short_term) > self.short_term_capacity:
            evicted = self.short_term.pop(0)
            # 被挤出的高权重记忆自动升入长期记忆
            if evicted.emotional_weight > 0.6 or evicted.is_pinned:
                self._push_long_term(evicted)

    def _push_long_term(self, entry: MemoryEntry):
        heapq.heappush(self.long_term, entry)
        if len(self.long_term) > self.long_term_capacity:
            # 剔除权重最低的（堆尾）
            self.long_term = sorted(self.long_term)[:self.long_term_capacity]

    def retrieve_context(self, top_n=10) -> list[MemoryEntry]:
        """检索当前最相关记忆，用于拼接LLM上下文"""
        recent   = self.short_term[-top_n:]
        critical = self.long_term[:top_n // 2]
        return recent + critical


# ================================================================
# 三、性格参数过滤器
# ================================================================

class PersonalityFilter:
    """
    将原始意图列表按性格参数重新加权，
    输出调整后的意图概率分布。
    """
    def __init__(self, params: PersonalityParams):
        self.params = params

    def apply(self, raw_intent_probs: dict[ActionIntent, float]) -> dict[ActionIntent, float]:
        p = self.params
        adjusted = {}
        for intent, prob in raw_intent_probs.items():
            weight = 1.0
            if intent == ActionIntent.RUB_LEG:  weight *= (1 + p.clingy)
            if intent == ActionIntent.HIDE:     weight *= (1 + p.timid)
            if intent == ActionIntent.FLEE:     weight *= (1 + p.timid)
            if intent == ActionIntent.EAT:      weight *= (1 + p.greedy)
            if intent == ActionIntent.PLAY:     weight *= (1 + p.playful)
            adjusted[intent] = prob * weight

        # 归一化
        total = sum(adjusted.values()) or 1.0
        return {k: v / total for k, v in adjusted.items()}


# ================================================================
# 四、意图决策模块（MDP + Embedding + Transformer 抽象）
# ================================================================

class IntentDecisionModule:
    """
    输入：当前状态感知 + 记忆上下文 + 性格参数
    输出：动作意图 + 情感/话题标签
    内部使用 MDP 奖励建模 + Transformer 编码状态序列
    （此处为接口抽象，实际模型调用在此插入）
    """
    def __init__(self, personality: PersonalityParams, filter: PersonalityFilter):
        self.personality = personality
        self.filter      = filter

    def decide(
        self,
        perception: dict,           # 当前环境感知（玩家动作、附近顾客等）
        memory_context: list[MemoryEntry],
    ) -> tuple[ActionIntent, EmotionTag]:

        # Step 1: 将感知 + 记忆序列化为状态向量（Embedding）
        state_embedding = self._encode_state(perception, memory_context)

        # Step 2: Transformer 推断原始意图概率
        raw_probs: dict[ActionIntent, float] = self._transformer_infer(state_embedding)

        # Step 3: 性格过滤器调整概率分布
        adjusted_probs = self.filter.apply(raw_probs)

        # Step 4: 采样最终意图（ε-greedy 或 softmax 采样）
        chosen_intent = self._sample(adjusted_probs)

        # Step 5: 推断对应的情感/话题标签
        emotion_tag = self._infer_emotion_tag(chosen_intent, perception)

        return chosen_intent, emotion_tag

    def _encode_state(self, perception, memory_context) -> list[float]:
        """伪实现：实际为 Embedding 模型调用"""
        # perception + memory → 向量表示
        return [0.0] * 512  # placeholder

    def _transformer_infer(self, embedding) -> dict[ActionIntent, float]:
        """伪实现：实际为本地轻量 Transformer 推理"""
        return {intent: 1.0 / len(ActionIntent) for intent in ActionIntent}

    def _sample(self, probs: dict) -> ActionIntent:
        """按概率分布采样"""
        import random
        actions = list(probs.keys())
        weights = list(probs.values())
        return random.choices(actions, weights=weights, k=1)[0]

    def _infer_emotion_tag(self, intent: ActionIntent, perception: dict) -> EmotionTag:
        """根据意图和当前感知推断情感标签"""
        emotion_map = {
            ActionIntent.RUB_LEG: EmotionTag("依赖",   "主人",   0.8),
            ActionIntent.HIDE:    EmotionTag("戒备",   "陌生人", 0.7),
            ActionIntent.MEOW:    EmotionTag("渴望",   "食物",   0.6),
            ActionIntent.FLEE:    EmotionTag("恐惧",   "创伤",   0.9),
            ActionIntent.PURR:    EmotionTag("满足",   "安全感", 0.85),
        }
        return emotion_map.get(intent, EmotionTag("平静", "日常", 0.3))


# ================================================================
# 五、轻量LLM — 猫语文字泡生成
# ================================================================

class CatSpeechGenerator:
    """
    输入：情感标签 + 记忆上下文 + 性格人设
    输出：猫语内心独白文本（展示为游戏内文字泡）
    """
    def __init__(self, personality: PersonalityParams):
        self.personality = personality

    def generate(self, emotion_tag: EmotionTag, memory_context: list[MemoryEntry]) -> str:
        system_prompt = self._build_system_prompt()
        user_prompt   = self._build_user_prompt(emotion_tag, memory_context)

        # 调用轻量 LLM（本地部署或云端小参数模型）
        response = self._call_llm(system_prompt, user_prompt)
        return response

    def _build_system_prompt(self) -> str:
        p = self.personality
        return (
            f"你是一只名叫【{p.name}】的猫。"
            f"你的性格：黏人度{p.clingy:.1f}，胆怯度{p.timid:.1f}，活泼度{p.playful:.1f}。"
            f"你的创伤背景：{', '.join(p.trauma_tags) or '无'}。"
            f"请用第一人称、简短的猫的视角，说出此刻内心的真实感受（20字以内，不要解释）。"
            f"你的语气和措辞必须始终与你的性格一致，绝不能前后矛盾。"
        )

    def _build_user_prompt(self, tag: EmotionTag, memories: list[MemoryEntry]) -> str:
        memory_text = "\n".join(
            f"- {m.event_description}" for m in memories[-5:]
        )
        return (
            f"当前情绪：{tag.emotion}（强度{tag.intensity:.1f}），话题聚焦：{tag.topic}。\n"
            f"近期记忆片段：\n{memory_text}\n"
            f"请输出此刻的内心独白："
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """伪实现：实际调用 LLM API / 本地推理"""
        return f"[LLM生成的猫语内心独白，基于上述prompt]"


# ================================================================
# 六、猫咪个体控制器（整合上述模块）
# ================================================================

class CatIndividualController:
    """
    每只猫独立实例化，负责：
    感知 → 记忆 → 性格过滤 → 意图决策 → 动作/猫语输出
    """
    def __init__(self, personality: PersonalityParams):
        self.personality   = personality
        self.memory        = MemoryModule()
        self.filter        = PersonalityFilter(personality)
        self.decision      = IntentDecisionModule(personality, self.filter)
        self.speech_gen    = CatSpeechGenerator(personality)
        self.trust_level   = 0.0   # 0~1，影响叙事状态

    def perceive_and_respond(
        self,
        player_action: PlayerAction,
        environment: dict,
    ) -> dict:
        """
        主处理循环，每次玩家交互触发一次。
        返回：动作意图 + 猫语文字 + 动画状态机指令
        """
        # 1. 感知构建
        perception = {
            "player_action": player_action,
            "environment":   environment,
            "trust_level":   self.trust_level,
        }

        # 2. 检索记忆
        memory_ctx = self.memory.retrieve_context()

        # 3. 意图决策
        intent, emotion_tag = self.decision.decide(perception, memory_ctx)

        # 4. 生成猫语
        speech = self.speech_gen.generate(emotion_tag, memory_ctx)

        # 5. 更新记忆
        new_memory = MemoryEntry(
            event_description=f"玩家{player_action.value}，我选择了{intent.value}",
            emotional_weight=emotion_tag.intensity,
            timestamp=environment.get("game_time", 0),
        )
        self.memory.add_memory(new_memory)

        # 6. 更新信任值（副作用传递给叙事控制流）
        trust_delta = self._calc_trust_delta(player_action, intent)
        self.trust_level = min(1.0, max(0.0, self.trust_level + trust_delta))

        return {
            "action_intent":    intent,
            "animation_cmd":    self._map_to_animation(intent),
            "cat_speech":       speech,
            "emotion_tag":      emotion_tag,
            "trust_level":      self.trust_level,
        }

    def _calc_trust_delta(self, action: PlayerAction, intent: ActionIntent) -> float:
        """简单规则：抚摸+猫主动蹭腿 → 信任上升"""
        if action == PlayerAction.PET and intent == ActionIntent.RUB_LEG:
            return 0.05
        if intent == ActionIntent.FLEE:
            return -0.03
        return 0.01

    def _map_to_animation(self, intent: ActionIntent) -> str:
        """将意图映射到动画状态机指令"""
        anim_map = {
            ActionIntent.RUB_LEG: "ANIM_RUB_LEG",
            ActionIntent.MEOW:    "ANIM_MEOW",
            ActionIntent.HIDE:    "ANIM_HIDE",
            ActionIntent.PURR:    "ANIM_PURR",
            ActionIntent.FLEE:    "ANIM_FLEE",
            ActionIntent.SLEEP:   "ANIM_SLEEP",
        }
        return anim_map.get(intent, "ANIM_IDLE")


# ================================================================
# 七、叙事控制流
# ================================================================

@dataclass
class StoryNode:
    """故事图谱中的单个节点"""
    node_id: str
    description: str
    unlock_conditions: dict         # e.g. {"trust_level": 0.5, "day": 10}
    next_nodes: list[str]           # 后继节点 ID
    emotional_tone: str             # "温暖", "哀伤", "转折"
    flags_to_set: list[str] = field(default_factory=list)


class StoryGraph:
    """预定义故事图谱（设计师配置）"""
    def __init__(self, nodes: list[StoryNode]):
        self.nodes: dict[str, StoryNode] = {n.node_id: n for n in nodes}
        self.root_node_id = nodes[0].node_id if nodes else None

    def get_node(self, node_id: str) -> Optional[StoryNode]:
        return self.nodes.get(node_id)

    def get_available_next_nodes(
        self,
        current_node_id: str,
        narrative_state: "NarrativeState",
    ) -> list[StoryNode]:
        current = self.get_node(current_node_id)
        if not current:
            return []
        available = []
        for nid in current.next_nodes:
            node = self.get_node(nid)
            if node and narrative_state.check_conditions(node.unlock_conditions):
                available.append(node)
        return available


@dataclass
class NarrativeState:
    """全局叙事状态追踪器"""
    current_node_id: str
    progress: float             # 0~1 总体进度
    trust_values: dict[str, float]   # cat_name → trust
    flags: set[str]             # 已触发的关键 flag
    emotional_tone: str         # 当前整体情感基调
    day: int = 0

    def check_conditions(self, conditions: dict) -> bool:
        """检查节点解锁条件"""
        for key, required_val in conditions.items():
            if key == "trust_level":
                if all(t < required_val for t in self.trust_values.values()):
                    return False
            elif key == "day":
                if self.day < required_val:
                    return False
            elif key == "flag":
                if required_val not in self.flags:
                    return False
        return True

    def apply_flags(self, flags: list[str]):
        self.flags.update(flags)


class ConsistencyFilter:
    """
    对LLM叙事输出进行一致性校验：
    事实一致性 / 人设一致性 / 情感基调一致性
    """
    def __init__(self, cat_personalities: dict[str, PersonalityParams]):
        self.personalities = cat_personalities

    def validate(self, generated_text: str, narrative_state: NarrativeState) -> tuple[bool, str]:
        """
        返回 (is_valid, reason)
        实际实现可用规则引擎 + 小分类模型
        """
        # 检查一：情感基调是否匹配
        if narrative_state.emotional_tone == "温暖" and "绝望" in generated_text:
            return False, "情感基调不匹配"

        # 检查二：人设关键词是否违反（示例规则）
        for cat_name, params in self.personalities.items():
            if cat_name in generated_text:
                if params.timid > 0.8 and "主动扑向陌生人" in generated_text:
                    return False, f"{cat_name} 性格违反"

        # 更多规则可扩展...
        return True, "通过"


class NarrativeLLMGenerator:
    """
    叙事 LLM：受控创意生成
    输入：故事节点 + 叙事状态 + 信任值
    输出：剧情对话 / 场景描述文本
    """
    def generate(
        self,
        node: StoryNode,
        narrative_state: NarrativeState,
        cat_personalities: dict[str, PersonalityParams],
    ) -> str:
        system_prompt = (
            f"你是《猫语心声》的叙事引擎。"
            f"当前情感基调：{node.emotional_tone}。"
            f"请严格基于以下故事节点和世界观，生成沉浸式的剧情描述，"
            f"不要引入任何节点外的新设定。"
        )
        user_prompt = (
            f"故事节点：{node.description}\n"
            f"当前进度：{narrative_state.progress:.0%}，"
            f"第{narrative_state.day}天，"
            f"信任值：{narrative_state.trust_values}\n"
            f"请生成此刻的剧情描述（100字以内）："
        )
        return self._call_llm(system_prompt, user_prompt)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """伪实现：实际调用大参数叙事 LLM"""
        return "[LLM生成的叙事剧情文本]"


class NarrativeController:
    """
    整体叙事控制器（全局）
    负责：故事图谱推进 + LLM生成 + 一致性过滤 + 状态更新
    """
    def __init__(
        self,
        story_graph: StoryGraph,
        cat_personalities: dict[str, PersonalityParams],
    ):
        self.story_graph    = story_graph
        self.llm_generator  = NarrativeLLMGenerator()
        self.filter         = ConsistencyFilter(cat_personalities)
        self.cat_personalities = cat_personalities

        # 初始化叙事状态
        self.state = NarrativeState(
            current_node_id  = story_graph.root_node_id,
            progress         = 0.0,
            trust_values     = {name: 0.0 for name in cat_personalities},
            flags            = set(),
            emotional_tone   = "温暖",
        )

    def sync_trust_from_cats(self, cat_controllers: dict[str, CatIndividualController]):
        """从各猫咪控制器同步信任值到叙事状态"""
        for name, ctrl in cat_controllers.items():
            self.state.trust_values[name] = ctrl.trust_level

    def try_advance_story(self) -> Optional[str]:
        """
        尝试推进故事节点。
        若满足解锁条件，生成下一段剧情并更新叙事状态。
        返回生成的剧情文本，或 None（条件未满足）
        """
        available_nodes = self.story_graph.get_available_next_nodes(
            self.state.current_node_id, self.state
        )
        if not available_nodes:
            return None

        next_node = available_nodes[0]   # 可扩展为多分支选择逻辑

        # LLM 生成剧情（带重试机制）
        MAX_RETRY = 3
        for attempt in range(MAX_RETRY):
            generated_text = self.llm_generator.generate(
                next_node, self.state, self.cat_personalities
            )
            is_valid, reason = self.filter.validate(generated_text, self.state)
            if is_valid:
                break
            print(f"[一致性校验失败 第{attempt+1}次] 原因：{reason}，重新生成...")
        else:
            # 超出重试次数，降级为预设文本
            generated_text = next_node.description

        # 更新叙事状态
        self.state.current_node_id = next_node.node_id
        self.state.progress = min(1.0, self.state.progress + 0.05)
        self.state.emotional_tone = next_node.emotional_tone
        self.state.apply_flags(next_node.flags_to_set)

        return generated_text


# ================================================================
# 八、主游戏循环（整合两条控制流）
# ================================================================

class CatCafeGame:
    """
    游戏主控，整合：
    - 多只猫咪的个体控制流
    - 全局叙事控制流
    - 玩家输入路由
    """
    def __init__(self, cat_configs: list[PersonalityParams], story_nodes: list[StoryNode]):
        # 实例化各猫咪控制器
        self.cat_controllers: dict[str, CatIndividualController] = {
            cfg.name: CatIndividualController(cfg) for cfg in cat_configs
        }

        # 实例化叙事控制器
        story_graph = StoryGraph(story_nodes)
        cat_personalities = {cfg.name: cfg for cfg in cat_configs}
        self.narrative = NarrativeController(story_graph, cat_personalities)

        self.game_time = 0

    def on_player_action(self, action: PlayerAction, target_cat: Optional[str], env: dict):
        """
        玩家触发动作的统一入口。
        路由到个体控制流 或 叙事控制流。
        """
        env["game_time"] = self.game_time

        # ── 路由判断 ──────────────────────────────────────────────
        if action == PlayerAction.STORY_NODE:
            # 关键叙事节点 → 叙事控制流
            self.narrative.sync_trust_from_cats(self.cat_controllers)
            story_text = self.narrative.try_advance_story()
            if story_text:
                self._display_narrative(story_text)

        elif target_cat and target_cat in self.cat_controllers:
            # 针对某只猫的具体互动 → 个体控制流
            ctrl = self.cat_controllers[target_cat]
            result = ctrl.perceive_and_respond(action, env)
            self._display_cat_response(target_cat, result)

            # 个体信任变化可能触发叙事推进
            self.narrative.sync_trust_from_cats(self.cat_controllers)
            story_text = self.narrative.try_advance_story()
            if story_text:
                self._display_narrative(story_text)

        self.game_time += 1

    def _display_cat_response(self, cat_name: str, result: dict):
        """游戏引擎展示层（渲染动画 + 文字泡）"""
        print(f"\n🐱 [{cat_name}] 动作：{result['action_intent'].value}")
        print(f"   动画指令：{result['animation_cmd']}")
        print(f"   💬 猫语：{result['cat_speech']}")
        print(f"   信任值：{result['trust_level']:.2f}")

    def _display_narrative(self, text: str):
        """游戏引擎展示层（渲染叙事剧情）"""
        print(f"\n📖 [剧情] {text}")


# ================================================================
# 九、辅助模块 — 离线一致性测试
# ================================================================

class OfflineConsistencyTester:
    """
    设计师工具：批量测试猫咪/叙事的一致性，
    验证 personality 参数与 LLM 输出是否符合预期。
    """
    def run_cat_speech_test(
        self,
        personality: PersonalityParams,
        test_cases: list[dict],
    ):
        ctrl = CatIndividualController(personality)
        print(f"\n=== 一致性测试：{personality.name} ===")
        for case in test_cases:
            action = case["action"]
            env    = case.get("env", {})
            result = ctrl.perceive_and_respond(action, env)
            print(f"  输入动作：{action.value}")
            print(f"  输出意图：{result['action_intent'].value}")
            print(f"  猫语：{result['cat_speech']}")
            print()

    def run_narrative_consistency_test(
        self,
        story_graph: StoryGraph,
        cat_personalities: dict[str, PersonalityParams],
        simulated_trust: dict[str, float],
    ):
        ctrl = NarrativeController(story_graph, cat_personalities)
        ctrl.state.trust_values = simulated_trust
        print("\n=== 叙事一致性测试 ===")
        for _ in range(5):
            text = ctrl.try_advance_story()
            if text:
                print(f"  节点：{ctrl.state.current_node_id}")
                print(f"  生成剧情：{text}\n")
            else:
                print("  无可用后继节点，测试结束。")
                break


# ================================================================
# 十、示例入口
# ================================================================

if __name__ == "__main__":
    # 配置3只原住民猫咪（设计师填写）
    cats = [
        PersonalityParams(
            name="橘墩",    clingy=0.9, timid=0.1, greedy=0.95, playful=0.7,
            trauma_tags=[], backstory="从小在猫咖长大，对人类充满信任。"
        ),
        PersonalityParams(
            name="霜降",    clingy=0.2, timid=0.85, greedy=0.3, playful=0.4,
            trauma_tags=["被遗弃", "曾受过伤"], backstory="流浪过两年，对陌生人极度戒备。"
        ),
        PersonalityParams(
            name="墨玉",    clingy=0.5, timid=0.3, greedy=0.5, playful=0.9,
            trauma_tags=[], backstory="喜欢恶作剧，总在最意外的时刻出现。"
        ),
    ]

    # 配置故事图谱（设计师填写）
    story_nodes = [
        StoryNode("开篇",    "玩家初次进入猫咖",         {"day": 0},  ["初识信任"], "温暖"),
        StoryNode("初识信任", "橘墩第一次主动蹭了你的腿", {"trust_level": 0.3, "day": 3}, ["霜降解锁"], "温暖"),
        StoryNode("霜降解锁", "霜降终于从柜子后探出了头", {"trust_level": 0.5, "day": 7}, ["结局预告"], "哀伤"),
        StoryNode("结局预告", "你意识到这里已经是它们的家", {}, [], "温暖"),
    ]

    # 启动游戏
    game = CatCafeGame(cats, story_nodes)

    # 模拟玩家操作序列
    game.on_player_action(PlayerAction.PET,  "橘墩", {"location": "沙发区"})
    game.on_player_action(PlayerAction.FEED, "霜降", {"food": "小鱼干"})
    game.on_player_action(PlayerAction.PLAY, "墨玉", {"toy": "逗猫棒"})
    game.on_player_action(PlayerAction.STORY_NODE, None, {})