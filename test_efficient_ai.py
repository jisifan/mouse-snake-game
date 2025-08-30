# -*- coding: utf-8 -*-
"""高效AI快速测试脚本"""
import sys
sys.path.append('.')
from snake_rl_env import SnakeRLEnv

def test_efficient_rewards():
    """测试新奖励系统"""
    print("=== 测试高效AI奖励系统 ===")
    
    env = SnakeRLEnv()
    obs1, obs2 = env.reset()
    
    print(f"初始: 蛇1得分={env.snake1_score}, 蛇2得分={env.snake2_score}")
    
    # 模拟快速吃食物场景
    for step in range(20):
        actions = (env.action_space.sample(), env.action_space.sample())
        (obs1, obs2), (r1, r2), (d1, d2), info = env.step(actions)
        
        if step == 5:
            env.snake1_score = 5  # 模拟到达5食物里程碑
            print("🎯 模拟蛇1达到5食物里程碑")
        
        if info['snake1_milestones']['5']:
            print(f"✅ 蛇1达成5食物里程碑! 奖励: {r1:.1f}")
            break
    
    env.close()
    print("测试完成!")

if __name__ == "__main__":
    test_efficient_rewards()
