# -*- coding: utf-8 -*-
"""
测试新的200分里程碑奖励系统
========================

验证强化学习环境的新奖励机制是否正常工作。
"""

import numpy as np
from snake_rl_env import SnakeRLEnv

def test_milestone_reward_system():
    """测试200分里程碑奖励系统"""
    print("=== 测试200分里程碑奖励系统 ===")
    
    env = SnakeRLEnv()
    obs1, obs2 = env.reset()
    
    print(f"初始状态:")
    print(f"- 蛇1得分: {env.snake1_score}")
    print(f"- 蛇2得分: {env.snake2_score}")
    print(f"- 里程碑是否达成: {env.milestone_200_reached}")
    print()
    
    # 模拟蛇1达到200分的情况
    print("模拟蛇1吃食物，逐步接近200分...")
    step_count = 0
    
    while env.snake1_score < 200 and step_count < 1000:
        # 随机动作
        action1 = env.action_space.sample()
        action2 = env.action_space.sample()
        
        # 手动让蛇1得分（模拟吃食物）
        if step_count % 10 == 0:  # 每10步让蛇1得10分
            env.snake1_score += 10
            print(f"步骤 {step_count}: 蛇1得分 = {env.snake1_score}")
        
        # 执行动作
        (obs1, obs2), (reward1, reward2), (done1, done2), info = env.step((action1, action2))
        
        # 检查是否达到里程碑
        if env.milestone_200_reached:
            print(f"\n里程碑达成！")
            print(f"- 获胜者: 蛇{env.winner_200_milestone}")
            print(f"- 蛇1奖励: {reward1}")
            print(f"- 蛇2奖励: {reward2}")
            print(f"- 蛇1最终得分: {env.snake1_score}")
            print(f"- 蛇2最终得分: {env.snake2_score}")
            break
            
        step_count += 1
    
    env.close()
    print("\n测试完成！")

def test_food_scoring_system():
    """测试食物得分系统"""
    print("\n=== 测试食物得分系统 ===")
    
    env = SnakeRLEnv()
    obs1, obs2 = env.reset()
    
    print("模拟蛇吃食物的得分机制...")
    
    # 模拟蛇1吃食物
    original_score1 = env.snake1_score
    env.snake1_score += 10  # 模拟吃一个食物
    
    # 模拟蛇2吃食物
    original_score2 = env.snake2_score  
    env.snake2_score += 20  # 模拟吃两个食物
    
    print(f"蛇1: {original_score1} -> {env.snake1_score} (+10)")
    print(f"蛇2: {original_score2} -> {env.snake2_score} (+20)")
    
    # 测试得分差距奖励
    action1 = 0  # 向上
    action2 = 1  # 向下
    
    (obs1, obs2), (reward1, reward2), (done1, done2), info = env.step((action1, action2))
    
    print(f"得分差距奖励测试:")
    print(f"- 蛇1奖励: {reward1}")
    print(f"- 蛇2奖励: {reward2}")
    print(f"- 得分差: {env.snake1_score - env.snake2_score}")
    
    env.close()

if __name__ == "__main__":
    test_milestone_reward_system()
    test_food_scoring_system()