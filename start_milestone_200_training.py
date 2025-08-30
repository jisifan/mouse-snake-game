# -*- coding: utf-8 -*-
"""
200分里程碑奖励训练系统
====================

专门训练带有200分里程碑奖励机制的强化学习AI。
这将产生更积极竞争、追求高分的AI模型。
"""

from snake_rl_trainer import SnakeRLTrainer
import os

def train_200_milestone_ai():
    """训练带有200分里程碑奖励的AI"""
    print("=== 200分里程碑奖励训练系统 ===")
    print("训练目标: 首先到达200分的AI获得巨大奖励")
    print("预期效果: AI将更积极地追求高分并与对手竞争")
    print()
    
    # 创建专门的模型保存路径
    milestone_model_path = "snake_rl_models_milestone_200"
    
    # 创建训练器
    trainer = SnakeRLTrainer(model_save_path=milestone_model_path)
    
    print(f"模型将保存到: {milestone_model_path}")
    print()
    
    # 开始训练
    print("开始200分里程碑奖励训练...")
    print("训练参数:")
    print("- 总训练步数: 300,000 (增加训练量以适应新奖励)")
    print("- 切换频率: 25,000 步")
    print("- 学习率: 2e-4 (稍微降低以获得更稳定的学习)")
    print()
    
    # 创建模型（使用稍微保守的参数）
    trainer.create_models(learning_rate=2e-4, n_steps=2048)
    
    # 开始交替训练
    trainer.train_alternating(
        total_timesteps=300000,  # 增加训练步数
        switch_frequency=25000   # 增加切换频率
    )
    
    print()
    print("=== 训练完成！评估新模型性能 ===")
    
    # 评估训练结果
    trainer.evaluate_models(n_eval_episodes=15, render=False)
    
    # 绘制训练统计
    trainer.plot_training_stats()
    
    print()
    print("训练总结:")
    print("1. 新的AI模型已保存到:", milestone_model_path)
    print("2. 这些模型被训练去追求200分里程碑")
    print("3. 预期它们会表现出更积极的竞争行为")
    print("4. 可以通过修改game_launcher.py来使用新模型")

def update_game_launcher_for_milestone_models():
    """更新游戏启动器以使用里程碑训练模型"""
    print()
    print("=== 更新游戏启动器建议 ===")
    print("要使用新的200分里程碑训练模型，请修改 game_launcher.py:")
    print()
    print("将这行:")
    print('  model_path = "snake_rl_models/agent2/model_60000.zip"')
    print()
    print("改为:")
    print('  model_path = "snake_rl_models_milestone_200/final_agent2.zip"')
    print()
    print("或者尝试不同的checkpoint模型以获得最佳性能")

if __name__ == "__main__":
    # 检查依赖
    try:
        import stable_baselines3
        import gymnasium
        
        train_200_milestone_ai()
        update_game_launcher_for_milestone_models()
        
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install stable-baselines3 gymnasium")