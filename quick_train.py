# -*- coding: utf-8 -*-
"""
快速强化学习训练脚本
===================

用于快速演示和测试强化学习系统的简化训练脚本。
如果需要完整训练，请运行snake_rl_trainer.py

使用方法:
1. pip install stable-baselines3 gym matplotlib
2. python quick_train.py

作者: Claude Code Assistant
"""

import os
import sys
import numpy as np

# 设置Windows控制台编码
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')

def check_dependencies():
    """检查必要的依赖库"""
    required_packages = [
        ('stable_baselines3', 'stable-baselines3'),
        ('gym', 'gym'),
        ('matplotlib', 'matplotlib')
    ]
    
    missing_packages = []
    
    for package_name, pip_name in required_packages:
        try:
            __import__(package_name)
            print(f"✓ {package_name} is available")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"✗ {package_name} not found")
    
    if missing_packages:
        print(f"\\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def quick_train():
    """快速训练演示"""
    print("=== 贪吃蛇强化学习快速训练 ===\\n")
    
    if not check_dependencies():
        return
    
    try:
        from snake_rl_trainer import SnakeRLTrainer
        
        print("Starting quick training (reduced timesteps for demo)...")
        
        # 创建训练器
        trainer = SnakeRLTrainer(model_save_path="snake_rl_models_quick")
        
        # 快速训练（较少的时间步数用于演示）
        print("\\n🚀 Starting quick training session...")
        trainer.train_alternating(total_timesteps=50000, switch_frequency=10000)
        
        print("\\n📊 Evaluating trained models...")
        trainer.evaluate_models(n_eval_episodes=5, render=False)
        
        print("\\n✅ Quick training completed!")
        print("\\nTrained models saved in: snake_rl_models_quick/")
        print("- final_agent1.zip")
        print("- final_agent2.zip")
        
        # 绘制训练统计
        trainer.plot_training_stats()
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        print("\\nTroubleshooting tips:")
        print("1. Make sure all dependencies are installed")
        print("2. Try running: pip install stable-baselines3 gym matplotlib")
        print("3. Check that Python version is 3.7+")
        return False

def create_random_models():
    """创建随机行为的模拟AI模型（用于测试集成）"""
    print("\\n🎲 Creating random baseline models for testing...")
    
    models_dir = "snake_rl_models_demo"
    os.makedirs(models_dir, exist_ok=True)
    
    # 创建一个简单的随机AI类
    random_ai_code = '''
"""
随机行为AI（用于测试集成）
"""
import random

class RandomSnakeAI:
    def __init__(self):
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7]  # 8个方向
    
    def predict(self, observation, deterministic=False):
        # 随机选择动作，但避免明显危险的动作
        if len(observation) >= 17:  # 检查危险度信息
            danger_scores = observation[8:16]  # 8个方向的危险度
            
            # 优先选择安全的方向
            safe_actions = []
            for i, danger in enumerate(danger_scores):
                if danger < 0.5:  # 不太危险
                    safe_actions.append(i)
            
            if safe_actions:
                action = random.choice(safe_actions)
            else:
                action = random.choice(self.actions)
        else:
            action = random.choice(self.actions)
        
        return action, None
    
    def save(self, path):
        # 模拟保存功能
        with open(path.replace('.zip', '_random.txt'), 'w') as f:
            f.write("Random AI model placeholder")
    
    @classmethod
    def load(cls, path, env=None):
        return cls()
'''
    
    # 保存随机AI代码
    with open(os.path.join(models_dir, 'random_ai.py'), 'w') as f:
        f.write(random_ai_code)
    
    # 创建占位符模型文件
    with open(os.path.join(models_dir, 'final_agent1_random.txt'), 'w') as f:
        f.write("Random AI model placeholder for agent 1")
    
    with open(os.path.join(models_dir, 'final_agent2_random.txt'), 'w') as f:
        f.write("Random AI model placeholder for agent 2")
    
    print(f"✅ Random baseline models created in: {models_dir}/")
    return models_dir

def main():
    """主函数"""
    print("欢迎使用贪吃蛇强化学习训练系统！\\n")
    
    choice = input("选择训练模式:\\n1. 完整强化学习训练（需要较长时间）\\n2. 快速演示训练\\n3. 创建随机基线模型（用于测试集成）\\n请选择 (1/2/3): ")
    
    if choice == '1':
        print("\\n启动完整训练...")
        if check_dependencies():
            try:
                from snake_rl_trainer import SnakeRLTrainer
                trainer = SnakeRLTrainer()
                trainer.train_alternating(total_timesteps=500000, switch_frequency=50000)
                trainer.evaluate_models(n_eval_episodes=20, render=True)
                trainer.plot_training_stats()
            except Exception as e:
                print(f"训练失败: {e}")
        
    elif choice == '2':
        print("\\n启动快速演示训练...")
        quick_train()
        
    elif choice == '3':
        print("\\n创建随机基线模型...")
        models_dir = create_random_models()
        print(f"\\n模型已创建，可以用于测试集成功能。")
        print(f"模型目录: {models_dir}")
        
    else:
        print("无效选择，退出。")

if __name__ == "__main__":
    main()