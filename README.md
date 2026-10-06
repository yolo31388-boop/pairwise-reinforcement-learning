# 游戏内强化学习与AI训练系统

## 项目描述

游戏内强化学习与AI训练系统的bug修复项目。核心模块 `ai/reinforcement_learning.py` 存在多个已知bug，需要修复后使测试全部通过。

## 已知问题

- 智能体训练完全没做AI不会学习
- 奖励函数完全没做奖励都是固定的
- 策略网络完全没做不能更新策略
- 价值网络完全没做不能评估状态价值
- 探索完全没做AI只会exploitation

## 运行测试

```bash
python -m pytest tests/test_reinforcement_learning.py -q
```

修复目标：所有 16 个测试全部通过。

## 技术要求

- 纯Python标准库，不引入第三方依赖
- 不修改测试文件
- 保持现有API接口不变
