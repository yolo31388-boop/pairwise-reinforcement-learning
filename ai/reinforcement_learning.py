"""
游戏内强化学习与AI训练系统 - 核心模块

包含：
- 智能体训练（环境交互 + 经验回放）
- 奖励函数（稀疏奖励 + 奖励塑形）
- 策略网络（策略梯度 + 探索策略）
- 价值网络（时序差分 + 优势函数）
- 探索（epsilon-greedy + 衰减）
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random
import time


@dataclass
class Config:
    pass


class ReinforcementLearning:
    """表格型 Actor-Critic 游戏 AI 训练系统。

    内置一维走廊环境：智能体从 0 号格子出发，走到最右端格子即成功。
    动作 0 = 向左移动，1 = 向右移动。
    """

    ACTIONS = (0, 1)  # 0=左, 1=右

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []

        # 超参数
        self.learning_rate = float(self.config.get("learning_rate", 0.1))
        self.gamma = float(self.config.get("gamma", 0.9))
        self.epsilon = float(self.config.get("epsilon_start", 1.0))
        self.epsilon_min = float(self.config.get("epsilon_min", 0.05))
        self.epsilon_decay = float(self.config.get("epsilon_decay", 0.95))

        # 环境
        self.env_size = int(self.config.get("env_size", 5))
        self.max_steps = int(self.config.get("max_steps", 50))
        self.agent_pos = 0

        # 经验回放缓冲区
        self.replay_buffer = deque(
            maxlen=int(self.config.get("buffer_size", 1000))
        )
        self.batch_size = int(self.config.get("batch_size", 8))

        # 策略网络：状态 -> 各动作偏好值（softmax 策略）
        self.policy_prefs: Dict[int, Dict[int, float]] = {}
        # 价值网络：状态 -> 状态价值 V(s)
        self.value_table: Dict[int, float] = {}

        self.rng = random.Random(self.config.get("seed"))

    # ------------------------------------------------------------------
    # 环境交互
    # ------------------------------------------------------------------
    def reset_env(self) -> int:
        """重置环境，返回初始状态。"""
        self.agent_pos = 0
        return self.agent_pos

    def env_step(self, action: int) -> Tuple[int, float, bool]:
        """环境推进一步，返回 (下一状态, 稀疏奖励, 是否结束)。"""
        if action == 1:
            self.agent_pos = min(self.agent_pos + 1, self.env_size - 1)
        else:
            self.agent_pos = max(self.agent_pos - 1, 0)
        done = self.agent_pos == self.env_size - 1
        reward = 1.0 if done else 0.0
        return self.agent_pos, reward, done

    # ------------------------------------------------------------------
    # 奖励函数：稀疏奖励 + 奖励塑形
    # ------------------------------------------------------------------
    def reward_function(
        self, state: int, action: int, next_state: int, done: bool
    ) -> float:
        """稀疏奖励：到达终点 +1，其余情况为 0。"""
        return 1.0 if done else 0.0

    def _potential(self, state: int) -> float:
        """势函数：离终点越近势能越高。"""
        goal = self.env_size - 1
        return -abs(goal - state) / max(goal, 1)

    def reward_shaping(
        self, state: int, action: int, next_state: int, done: bool
    ) -> float:
        """基于势能的奖励塑形 F = gamma * Phi(s') - Phi(s)。"""
        return self.gamma * self._potential(next_state) - self._potential(state)

    def shaped_reward(
        self, state: int, action: int, next_state: int, done: bool
    ) -> float:
        """塑形奖励 = 稀疏奖励 + 奖励塑形，用于引导学习。"""
        return self.reward_function(state, action, next_state, done) + \
            self.reward_shaping(state, action, next_state, done)

    # ------------------------------------------------------------------
    # 策略网络：softmax 策略 + 策略梯度
    # ------------------------------------------------------------------
    def _prefs(self, state: int) -> Dict[int, float]:
        if state not in self.policy_prefs:
            self.policy_prefs[state] = {a: 0.0 for a in self.ACTIONS}
        return self.policy_prefs[state]

    def policy_network(self, state: int) -> Dict[int, float]:
        """策略网络前向传播，返回各动作的概率分布（softmax）。"""
        prefs = self._prefs(state)
        max_pref = max(prefs.values())
        exps = {a: math.exp(p - max_pref) for a, p in prefs.items()}
        total = sum(exps.values())
        return {a: e / total for a, e in exps.items()}

    def policy_gradient_update(
        self, state: int, action: int, advantage: float
    ) -> None:
        """策略梯度更新：theta += lr * advantage * grad log pi(a|s)。"""
        prefs = self._prefs(state)
        probs = self.policy_network(state)
        for a in self.ACTIONS:
            indicator = 1.0 if a == action else 0.0
            prefs[a] += self.learning_rate * advantage * (indicator - probs[a])

    # ------------------------------------------------------------------
    # 价值网络：时序差分 + 优势函数
    # ------------------------------------------------------------------
    def value_network(self, state: int) -> float:
        """价值网络前向传播，返回状态价值 V(s)。"""
        return self.value_table.get(state, 0.0)

    def td_update(
        self, state: int, reward: float, next_state: int, done: bool
    ) -> float:
        """时序差分 TD(0) 更新：V(s) += lr * (r + gamma * V(s') - V(s))。"""
        target = reward + (
            0.0 if done else self.gamma * self.value_network(next_state)
        )
        td_error = target - self.value_network(state)
        self.value_table[state] = self.value_network(state) + \
            self.learning_rate * td_error
        return td_error

    def advantage(
        self, state: int, action: int, reward: float, next_state: int,
        done: bool
    ) -> float:
        """优势函数估计：A = r + gamma * V(s') - V(s)（即 TD 误差）。"""
        target = reward + (
            0.0 if done else self.gamma * self.value_network(next_state)
        )
        return target - self.value_network(state)

    # ------------------------------------------------------------------
    # 探索：epsilon-greedy + 衰减
    # ------------------------------------------------------------------
    def exploration(self, state: int) -> int:
        """epsilon-greedy：以 epsilon 概率随机探索，否则利用当前策略。"""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(self.ACTIONS))
        probs = self.policy_network(state)
        return max(probs, key=probs.get)

    def decay_epsilon(self) -> float:
        """探索率随时间衰减，且不低于 epsilon_min。"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    # ------------------------------------------------------------------
    # 经验回放
    # ------------------------------------------------------------------
    def store_experience(
        self, state: int, action: int, reward: float, next_state: int,
        done: bool
    ) -> None:
        """把与环境交互得到的经验存入回放缓冲区。"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self, batch_size: Optional[int] = None) -> int:
        """从回放缓冲区采样历史经验重用，更新价值与策略网络。"""
        size = batch_size or self.batch_size
        if not self.replay_buffer:
            return 0
        batch = self.rng.sample(
            list(self.replay_buffer), min(size, len(self.replay_buffer))
        )
        for state, action, reward, next_state, done in batch:
            adv = self.advantage(state, action, reward, next_state, done)
            self.td_update(state, reward, next_state, done)
            self.policy_gradient_update(state, action, adv)
        return len(batch)

    # ------------------------------------------------------------------
    # 智能体训练：环境交互 + 经验回放
    # ------------------------------------------------------------------
    def train_agent(self, episodes: int = 100) -> Dict:
        """训练智能体：与环境交互收集经验，并通过经验回放学习。"""
        episode_rewards: List[float] = []
        successes = 0
        for _ in range(episodes):
            state = self.reset_env()
            total_reward = 0.0
            for _ in range(self.max_steps):
                action = self.exploration(state)
                next_state, _, done = self.env_step(action)
                reward = self.shaped_reward(state, action, next_state, done)
                self.store_experience(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                if done:
                    successes += 1
                    break
            self.replay()
            self.decay_epsilon()
            episode_rewards.append(total_reward)
        stats = {
            "episodes": episodes,
            "episode_rewards": episode_rewards,
            "success_rate": successes / max(episodes, 1),
            "epsilon": self.epsilon,
            "experiences": len(self.replay_buffer),
        }
        self._history.append(stats)
        return stats

    # ------------------------------------------------------------------
    # 框架接口
    # ------------------------------------------------------------------
    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self.policy_prefs = {}
        self.value_table = {}
        self.replay_buffer.clear()
        self.epsilon = float(self.config.get("epsilon_start", 1.0))
        self.agent_pos = 0
