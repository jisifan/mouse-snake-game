# -*- coding: utf-8 -*-
"""
强化学习AI训练启动器
===================

直接启动双AI对抗训练，无需交互输入
"""

import os
import sys

def start_training():
    """启动强化学习训练"""
    print("=" * 60)
    print("贪吃蛇强化学习AI训练系统启动")
    print("=" * 60)
    
    try:
        # 导入训练系统
        from snake_rl_trainer import SnakeRLTrainer
        
        print("\n训练配置:")
        print("- 训练步数: 200,000")
        print("- 切换频率: 20,000 (每2万步切换训练智能体)")
        print("- 算法: PPO (Proximal Policy Optimization)")
        print("- 环境: 双AI对抗贪吃蛇")
        print("- 模型保存: snake_rl_models/")
        
        print("\n开始训练...")
        
        # 创建训练器
        trainer = SnakeRLTrainer(model_save_path="snake_rl_models")
        
        print("\n执行交替训练...")
        # 开始交替训练 - 使用适中的训练参数
        trainer.train_alternating(total_timesteps=200000, switch_frequency=20000)
        
        print("\n评估训练结果...")
        # 评估训练好的模型
        trainer.evaluate_models(n_eval_episodes=10, render=False)
        
        print("\n生成训练统计图...")
        # 绘制训练统计
        trainer.plot_training_stats()
        
        print("\n训练完成!")
        print("模型已保存到: snake_rl_models/")
        print("- final_agent1.zip")
        print("- final_agent2.zip")
        print("- 以及各个检查点模型")
        
        return True
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保安装了所有依赖: pip install stable-baselines3 gym matplotlib")
        return False
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = start_training()
    if not success:
        sys.exit(1)