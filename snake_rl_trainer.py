# -*- coding: utf-8 -*-
"""
贪吃蛇多智能体强化学习训练系统
===================================

使用Stable-Baselines3实现双AI自对弈训练系统，包括：
- 多智能体环境包装器
- 自对弈训练机制
- 模型保存和评估
- 训练进度监控

作者: Claude Code Assistant
依赖: stable-baselines3, gym, numpy
"""

import gymnasium as gym
import numpy as np
import os
from typing import Dict, Tuple, Any, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from snake_rl_env import SnakeRLEnv
import matplotlib.pyplot as plt
from collections import deque

class MultiAgentWrapper(gym.Env):
    """
    多智能体环境包装器
    
    将双智能体环境包装为单智能体环境，用于与Stable-Baselines3兼容。
    通过轮流训练和自对弈机制实现双AI的协同进化。
    """
    
    def __init__(self, base_env: SnakeRLEnv, agent_id: int = 1):
        """
        初始化包装器
        
        参数:
            base_env: 基础的双智能体环境
            agent_id: 当前训练的智能体ID (1或2)
        """
        super().__init__()
        self.env = base_env
        self.agent_id = agent_id
        self.opponent_agent = None  # 对手智能体（用于自对弈）
        
        # 继承环境的动作和观察空间
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        
        # 统计数据
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rate = deque(maxlen=100)  # 最近100场的胜率
        
    def set_opponent(self, opponent_model):
        """设置对手模型（用于自对弈）"""
        self.opponent_agent = opponent_model
        
    def reset(self, **kwargs):
        """重置环境"""
        obs1, obs2 = self.env.reset()
        my_obs = obs1 if self.agent_id == 1 else obs2
        info = {}
        return my_obs, info
    
    def step(self, action):
        """执行一步动作"""
        # 获取对手动作
        if self.opponent_agent is not None:
            # 使用训练好的对手
            opponent_obs = self._get_opponent_obs()
            opponent_action, _ = self.opponent_agent.predict(opponent_obs, deterministic=False)
        else:
            # 随机对手
            opponent_action = self.action_space.sample()
        
        # 组合双方动作
        if self.agent_id == 1:
            actions = (action, opponent_action)
        else:
            actions = (opponent_action, action)
        
        # 执行环境步骤
        observations, rewards, dones, info = self.env.step(actions)
        
        # 返回当前智能体的结果
        my_obs = observations[0] if self.agent_id == 1 else observations[1]
        my_reward = rewards[0] if self.agent_id == 1 else rewards[1]
        my_done = dones[0] if self.agent_id == 1 else dones[1]
        
        # 更新统计数据
        if my_done:
            opponent_done = dones[1] if self.agent_id == 1 else dones[0]
            if not opponent_done and my_done:
                self.win_rate.append(0)  # 输了
            elif opponent_done and not my_done:
                self.win_rate.append(1)  # 赢了
            else:
                self.win_rate.append(0.5)  # 平局
        
        return my_obs, my_reward, my_done, False, info
    
    def _get_opponent_obs(self):
        """获取对手的观察（需要从环境状态构建）"""
        # 这里需要根据当前环境状态为对手构建观察
        # 为简化，我们使用当前观察的反向版本
        obs1, obs2 = self.env._get_observation(1), self.env._get_observation(2)
        return obs2 if self.agent_id == 1 else obs1
    
    def render(self, mode='human'):
        """渲染环境"""
        return self.env.render(mode)
    
    def close(self):
        """关闭环境"""
        self.env.close()
    
    def get_win_rate(self):
        """获取最近的胜率"""
        if len(self.win_rate) == 0:
            return 0.5
        return np.mean(self.win_rate)

class TrainingCallback(BaseCallback):
    """
    训练回调函数
    
    用于监控训练进度、保存模型检查点、记录统计数据。
    """
    
    def __init__(self, save_path: str, save_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        # 每隔一定步数保存模型
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f'model_{self.n_calls}.zip')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls} to {model_path}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        # 计算最近100个episode的平均奖励
        if len(self.episode_rewards) >= 100:
            mean_reward = np.mean(self.episode_rewards[-100:])
            
            # 如果性能提升，保存最佳模型
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = os.path.join(self.save_path, 'best_model.zip')
                self.model.save(best_model_path)
                if self.verbose > 0:
                    print(f"New best model saved! Mean reward: {mean_reward:.2f}")

class SnakeRLTrainer:
    """
    贪吃蛇强化学习训练器
    
    管理整个训练流程，包括环境创建、模型训练、自对弈等。
    """
    
    def __init__(self, model_save_path: str = "models", window_size: int = 600):
        """
        初始化训练器
        
        参数:
            model_save_path: 模型保存路径
            window_size: 游戏窗口大小
        """
        self.model_save_path = model_save_path
        self.window_size = window_size
        
        # 创建基础环境
        self.base_env = SnakeRLEnv(window_size=window_size)
        
        # 创建两个智能体的训练环境
        self.agent1_env = MultiAgentWrapper(self.base_env, agent_id=1)
        self.agent2_env = MultiAgentWrapper(self.base_env, agent_id=2)
        
        # 初始化模型
        self.agent1_model = None
        self.agent2_model = None
        
        # 训练统计
        self.training_stats = {
            'agent1_rewards': [],
            'agent2_rewards': [],
            'agent1_win_rates': [],
            'agent2_win_rates': [],
            'episodes': []
        }
        
        print(f"SnakeRLTrainer initialized. Models will be saved to: {model_save_path}")
    
    def create_models(self, learning_rate: float = 3e-4, n_steps: int = 2048):
        """
        创建PPO模型
        
        参数:
            learning_rate: 学习率
            n_steps: 每次更新的步数
        """
        # PPO超参数
        ppo_params = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'verbose': 1
        }
        
        # 创建两个智能体模型
        self.agent1_model = PPO('MlpPolicy', self.agent1_env, **ppo_params)
        self.agent2_model = PPO('MlpPolicy', self.agent2_env, **ppo_params)
        
        print("PPO models created for both agents")
    
    def train_alternating(self, total_timesteps: int = 100000, switch_frequency: int = 10000):
        """
        交替训练两个智能体
        
        参数:
            total_timesteps: 总训练步数
            switch_frequency: 切换智能体的频率
        """
        if self.agent1_model is None or self.agent2_model is None:
            self.create_models()
        
        print(f"Starting alternating training for {total_timesteps} timesteps")
        
        # 创建回调函数
        callback1 = TrainingCallback(os.path.join(self.model_save_path, 'agent1'), 
                                   save_freq=switch_frequency)
        callback2 = TrainingCallback(os.path.join(self.model_save_path, 'agent2'), 
                                   save_freq=switch_frequency)
        
        trained_steps = 0
        current_agent = 1
        
        while trained_steps < total_timesteps:
            steps_to_train = min(switch_frequency, total_timesteps - trained_steps)
            
            if current_agent == 1:
                print(f"\\nTraining Agent 1 for {steps_to_train} steps...")
                # 设置Agent1的对手为Agent2
                self.agent1_env.set_opponent(self.agent2_model)
                self.agent1_model.learn(total_timesteps=steps_to_train, callback=callback1)
                
                # 记录统计数据
                win_rate = self.agent1_env.get_win_rate()
                self.training_stats['agent1_win_rates'].append(win_rate)
                print(f"Agent 1 win rate: {win_rate:.3f}")
                
                current_agent = 2
            else:
                print(f"\\nTraining Agent 2 for {steps_to_train} steps...")
                # 设置Agent2的对手为Agent1
                self.agent2_env.set_opponent(self.agent1_model)
                self.agent2_model.learn(total_timesteps=steps_to_train, callback=callback2)
                
                # 记录统计数据
                win_rate = self.agent2_env.get_win_rate()
                self.training_stats['agent2_win_rates'].append(win_rate)
                print(f"Agent 2 win rate: {win_rate:.3f}")
                
                current_agent = 1
            
            trained_steps += steps_to_train
            print(f"Progress: {trained_steps}/{total_timesteps} ({trained_steps/total_timesteps*100:.1f}%)")
        
        print("\\nTraining completed!")
        self.save_final_models()
    
    def save_final_models(self):
        """保存最终训练好的模型"""
        if self.agent1_model:
            agent1_path = os.path.join(self.model_save_path, 'final_agent1.zip')
            self.agent1_model.save(agent1_path)
            print(f"Final Agent 1 model saved to: {agent1_path}")
        
        if self.agent2_model:
            agent2_path = os.path.join(self.model_save_path, 'final_agent2.zip')
            self.agent2_model.save(agent2_path)
            print(f"Final Agent 2 model saved to: {agent2_path}")
    
    def load_models(self, agent1_path: Optional[str] = None, agent2_path: Optional[str] = None):
        """
        加载预训练模型
        
        参数:
            agent1_path: Agent1模型路径
            agent2_path: Agent2模型路径
        """
        if agent1_path and os.path.exists(agent1_path):
            self.agent1_model = PPO.load(agent1_path, env=self.agent1_env)
            print(f"Agent 1 model loaded from: {agent1_path}")
        
        if agent2_path and os.path.exists(agent2_path):
            self.agent2_model = PPO.load(agent2_path, env=self.agent2_env)
            print(f"Agent 2 model loaded from: {agent2_path}")
    
    def evaluate_models(self, n_eval_episodes: int = 10, render: bool = True):
        """
        评估模型性能
        
        参数:
            n_eval_episodes: 评估的episode数量
            render: 是否渲染
        """
        if self.agent1_model is None or self.agent2_model is None:
            print("Models not loaded. Please train or load models first.")
            return
        
        print(f"\\nEvaluating models over {n_eval_episodes} episodes...")
        
        agent1_wins = 0
        agent2_wins = 0
        draws = 0
        
        for episode in range(n_eval_episodes):
            obs1, obs2 = self.base_env.reset()
            done = False
            
            while not done:
                # 获取两个智能体的动作
                action1, _ = self.agent1_model.predict(obs1, deterministic=True)
                action2, _ = self.agent2_model.predict(obs2, deterministic=True)
                
                # 执行动作
                (obs1, obs2), (reward1, reward2), (done1, done2), info = self.base_env.step((action1, action2))
                
                done = done1 or done2
                
                if render and episode < 3:  # 只渲染前几个episode
                    self.base_env.render()
                    import time
                    time.sleep(0.05)  # 减慢速度以便观察
                    
            # 统计胜负
            if done1 and not done2:
                agent2_wins += 1
            elif done2 and not done1:
                agent1_wins += 1
            else:
                draws += 1
            
            print(f"Episode {episode + 1}: Agent1={'Win' if done2 and not done1 else 'Loss' if done1 and not done2 else 'Draw'}")
        
        # 输出统计结果
        print(f"\\nEvaluation Results:")
        print(f"Agent 1 wins: {agent1_wins}/{n_eval_episodes} ({agent1_wins/n_eval_episodes*100:.1f}%)")
        print(f"Agent 2 wins: {agent2_wins}/{n_eval_episodes} ({agent2_wins/n_eval_episodes*100:.1f}%)")
        print(f"Draws: {draws}/{n_eval_episodes} ({draws/n_eval_episodes*100:.1f}%)")
        
        self.base_env.close()
    
    def plot_training_stats(self):
        """绘制训练统计图"""
        if not self.training_stats['agent1_win_rates']:
            print("No training statistics available.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 胜率变化
        plt.subplot(2, 1, 1)
        episodes = range(len(self.training_stats['agent1_win_rates']))
        plt.plot(episodes, self.training_stats['agent1_win_rates'], label='Agent 1 Win Rate', alpha=0.7)
        
        episodes = range(len(self.training_stats['agent2_win_rates']))
        plt.plot(episodes, self.training_stats['agent2_win_rates'], label='Agent 2 Win Rate', alpha=0.7)
        
        plt.xlabel('Training Iteration')
        plt.ylabel('Win Rate')
        plt.title('Agent Win Rates During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'training_stats.png'))
        plt.show()
        
        print(f"Training statistics plot saved to: {os.path.join(self.model_save_path, 'training_stats.png')}")

def main():
    """主训练函数"""
    print("=== 贪吃蛇强化学习训练系统 ===\\n")
    
    # 创建训练器
    trainer = SnakeRLTrainer(model_save_path="snake_rl_models")
    
    # 开始交替训练
    trainer.train_alternating(total_timesteps=200000, switch_frequency=20000)
    
    # 评估模型
    trainer.evaluate_models(n_eval_episodes=10, render=True)
    
    # 绘制训练统计
    trainer.plot_training_stats()

if __name__ == "__main__":
    main()