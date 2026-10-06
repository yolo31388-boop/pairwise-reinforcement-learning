"""
游戏内强化学习与AI训练系统 - 验收测试
运行方式：python -m pytest tests/test_reinforcement_learning.py -q
共 16 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.reinforcement_learning import ReinforcementLearning


class TestReinforcementLearning:
    def setup_method(self):
        self.system = ReinforcementLearning({"seed": 42})

    # ---- 智能体训练：环境交互 + 经验回放 ----

    def test_case_01(self):
        """train_agent 可运行并返回训练统计信息"""
        stats = self.system.train_agent(episodes=10)
        assert stats["episodes"] == 10
        assert len(stats["episode_rewards"]) == 10
        assert "success_rate" in stats

    def test_case_02(self):
        """训练过程中与环境交互并收集经验"""
        assert len(self.system.replay_buffer) == 0
        self.system.train_agent(episodes=5)
        assert len(self.system.replay_buffer) > 0
        state, action, reward, next_state, done = self.system.replay_buffer[0]
        assert isinstance(state, int)
        assert action in self.system.ACTIONS
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_case_03(self):
        """经验回放：重用历史经验更新网络"""
        for i in range(20):
            self.system.store_experience(0, 1, 0.5, 1, False)
        before = self.system.value_network(0)
        updated = self.system.replay(batch_size=8)
        assert updated == 8
        assert self.system.value_network(0) != before

    def test_case_04(self):
        """回放缓冲区容量受限，旧经验被挤出"""
        system = ReinforcementLearning({"seed": 1, "buffer_size": 10})
        for i in range(25):
            system.store_experience(i, 1, 0.0, i + 1, False)
        assert len(system.replay_buffer) == 10
        assert system.replay_buffer[-1][0] == 24

    def test_case_05(self):
        """训练后 AI 确实学会任务（成功率提升，不再是新手水平）"""
        stats = self.system.train_agent(episodes=200)
        assert stats["success_rate"] > 0.8
        late = stats["episode_rewards"][-20:]
        assert sum(1 for r in late if r > 0.5) >= 16

    # ---- 奖励函数：稀疏奖励 + 奖励塑形 ----

    def test_case_06(self):
        """稀疏奖励：到达终点 +1，其余状态为 0"""
        goal = self.system.env_size - 1
        assert self.system.reward_function(goal - 1, 1, goal, True) == 1.0
        assert self.system.reward_function(0, 1, 1, False) == 0.0
        assert self.system.reward_function(1, 0, 0, False) == 0.0

    def test_case_07(self):
        """奖励塑形：朝终点移动获得正向塑形，远离终点获得负向塑形"""
        toward = self.system.reward_shaping(1, 1, 2, False)
        away = self.system.reward_shaping(2, 0, 1, False)
        assert toward > 0
        assert away < 0
        expected = self.system.gamma * self.system._potential(2) \
            - self.system._potential(1)
        assert toward == pytest.approx(expected)

    def test_case_08(self):
        """塑形奖励 = 稀疏奖励 + 奖励塑形，能引导学习"""
        shaped = self.system.shaped_reward(1, 1, 2, False)
        sparse = self.system.reward_function(1, 1, 2, False)
        shaping = self.system.reward_shaping(1, 1, 2, False)
        assert shaped == pytest.approx(sparse + shaping)
        assert shaped > sparse

    # ---- 策略网络：策略梯度 + 探索策略 ----

    def test_case_09(self):
        """策略网络输出合法的概率分布"""
        probs = self.system.policy_network(0)
        assert set(probs.keys()) == set(self.system.ACTIONS)
        assert sum(probs.values()) == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in probs.values())

    def test_case_10(self):
        """策略梯度：正优势提升所选动作概率，负优势降低"""
        before = self.system.policy_network(0)[1]
        self.system.policy_gradient_update(0, 1, advantage=1.0)
        assert self.system.policy_network(0)[1] > before
        after = self.system.policy_network(0)[1]
        self.system.policy_gradient_update(0, 1, advantage=-5.0)
        assert self.system.policy_network(0)[1] < after

    def test_case_11(self):
        """策略网络包含探索策略：训练后利用时偏向更优动作"""
        self.system.train_agent(episodes=200)
        self.system.epsilon = 0.0
        actions = {self.system.exploration(0) for _ in range(10)}
        assert actions == {1}  # 在起点应坚定选择向右

    # ---- 价值网络：时序差分 + 优势函数 ----

    def test_case_12(self):
        """价值网络能评估状态价值"""
        assert self.system.value_network(0) == 0.0
        self.system.value_table[0] = 0.7
        assert self.system.value_network(0) == pytest.approx(0.7)

    def test_case_13(self):
        """时序差分：TD(0) 更新使 V(s) 向目标收敛"""
        td_error = self.system.td_update(0, reward=1.0, next_state=1, done=True)
        assert td_error == pytest.approx(1.0)
        assert self.system.value_network(0) == pytest.approx(
            self.system.learning_rate * 1.0
        )
        for _ in range(200):
            self.system.td_update(0, reward=1.0, next_state=1, done=True)
        assert self.system.value_network(0) == pytest.approx(1.0, abs=1e-3)

    def test_case_14(self):
        """优势函数：A = r + gamma * V(s') - V(s)"""
        self.system.value_table[0] = 0.5
        self.system.value_table[1] = 1.0
        adv = self.system.advantage(0, 1, reward=0.2, next_state=1, done=False)
        expected = 0.2 + self.system.gamma * 1.0 - 0.5
        assert adv == pytest.approx(expected)
        adv_done = self.system.advantage(0, 1, reward=1.0, next_state=1, done=True)
        assert adv_done == pytest.approx(1.0 - 0.5)

    # ---- 探索：epsilon-greedy + 衰减 ----

    def test_case_15(self):
        """epsilon-greedy：epsilon=1 纯探索，epsilon=0 纯利用"""
        self.system.epsilon = 1.0
        actions = {self.system.exploration(0) for _ in range(50)}
        assert actions == set(self.system.ACTIONS)  # 探索覆盖所有动作
        self.system.epsilon = 0.0
        self.system.policy_prefs[0] = {0: 0.0, 1: 2.0}
        greedy = {self.system.exploration(0) for _ in range(20)}
        assert greedy == {1}

    def test_case_16(self):
        """探索率随时间衰减且不低于下限"""
        start = self.system.epsilon
        self.system.decay_epsilon()
        assert self.system.epsilon == pytest.approx(start * self.system.epsilon_decay)
        for _ in range(1000):
            self.system.decay_epsilon()
        assert self.system.epsilon == pytest.approx(self.system.epsilon_min)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
