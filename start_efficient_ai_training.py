# -*- coding: utf-8 -*-
"""
高效AI训练系统 - 递进式里程碑版本
==============================

专门训练积极高效、快速吃食物的强化学习AI。
采用递进式里程碑奖励和效率导向机制。
"""

from snake_rl_trainer import SnakeRLTrainer
import os

def train_efficient_ai():
    """训练高效积极的AI"""
    print("=== 高效AI训练系统启动 ===")
    print()
    print("新奖励机制:")
    print("递进式里程碑: 5食物(+200) → 10食物(+300) → 20食物(+500)")
    print("大幅食物奖励: 每个食物+50分")
    print("移动效率: 接近食物+5分，远离-3分") 
    print("无效循环惩罚: 原地打转-10分")
    print("竞争激励: 领先奖励，追赶动机")
    print()
    
    # 创建专门的模型保存路径
    efficient_model_path = "snake_rl_models_efficient"
    
    # 创建训练器
    trainer = SnakeRLTrainer(model_save_path=efficient_model_path)
    
    print(f"模型将保存到: {efficient_model_path}")
    print()
    
    # 开始训练
    print("开始高效AI训练...")
    print("训练配置:")
    print("- 总训练步数: 200,000 (专注质量)")
    print("- 切换频率: 20,000 步 (加快学习)")
    print("- 学习率: 3e-4 (提高学习效率)")
    print("- 目标: 训练快速吃食物、高效移动的积极AI")
    print()
    
    # 创建模型（使用优化参数）
    trainer.create_models(learning_rate=3e-4, n_steps=2048)
    
    # 开始交替训练
    trainer.train_alternating(
        total_timesteps=200000,  # 专注质量训练
        switch_frequency=20000   # 更频繁的切换
    )
    
    print()
    print("=== 训练完成！评估高效AI ===")
    
    # 评估训练结果
    trainer.evaluate_models(n_eval_episodes=12, render=False)
    
    # 绘制训练统计
    trainer.plot_training_stats()
    
    print()
    print("训练总结:")
    print("1. 高效AI模型已保存:", efficient_model_path)
    print("2. 这些AI被训练成:")
    print("   - 积极快速吃食物")
    print("   - 高效直线移动") 
    print("   - 避免无效循环")
    print("   - 激烈竞争对战")
    print("3. 预期表现: 大幅减少无效移动，快速成长")

def update_game_for_efficient_ai():
    """提供游戏更新建议"""
    print()
    print("=== 使用高效AI模型建议 ===")
    print()
    print("要使用新训练的高效AI模型，建议尝试以下模型:")
    print()
    print("最新模型 (推荐):")
    print('   model_path = "snake_rl_models_efficient/final_agent2.zip"')
    print()
    print("中期checkpoint:")
    print('   model_path = "snake_rl_models_efficient/agent2/model_120000.zip"')
    print()
    print("早期checkpoint:")
    print('   model_path = "snake_rl_models_efficient/agent2/model_80000.zip"')
    print()
    print("建议测试不同checkpoint找到最佳性能的模型！")

def create_test_script():
    """创建快速测试脚本"""
    test_script = '''# -*- coding: utf-8 -*-
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
'''
    
    with open("test_efficient_ai.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    print("已创建快速测试脚本: test_efficient_ai.py")

if __name__ == "__main__":
    # 检查依赖
    try:
        import stable_baselines3
        import gymnasium
        
        print("依赖检查通过")
        
        # 创建测试脚本
        create_test_script()
        
        # 开始训练
        train_efficient_ai()
        
        # 提供使用建议
        update_game_for_efficient_ai()
        
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install stable-baselines3 gymnasium")