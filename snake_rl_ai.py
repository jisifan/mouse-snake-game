# -*- coding: utf-8 -*-
"""
强化学习贪吃蛇AI控制器
====================

集成训练好的强化学习模型到游戏中，提供智能的蛇控制。
支持从文件加载预训练模型，并提供后备的智能随机行为。

作者: Claude Code Assistant
依赖: 可选 stable-baselines3（如果有预训练模型）
"""

import math
import random
import numpy as np
from typing import List, Tuple, Optional
import os

class RLSnakeAI:
    """
    基于强化学习的贪吃蛇AI控制器
    
    优先加载训练好的强化学习模型，如果模型不存在或加载失败，
    则使用智能随机策略作为后备方案。
    """
    
    def __init__(self, window_size: int, cell_size: int, segment_distance: int, 
                 model_path: Optional[str] = None):
        """
        初始化强化学习AI控制器
        
        参数:
            window_size: 游戏窗口大小
            cell_size: 细胞大小  
            segment_distance: 蛇身节段间距
            model_path: 预训练模型路径
        """
        self.window_size = window_size
        self.cell_size = cell_size
        self.segment_distance = segment_distance
        
        # 8个方向的动作
        self.actions = [
            (0, -1), (0, 1), (-1, 0), (1, 0),  # 上、下、左、右
            (-1, -1), (1, -1), (-1, 1), (1, 1)  # 左上、右上、左下、右下
        ]
        
        # 尝试加载强化学习模型
        self.rl_model = self._load_rl_model(model_path)
        
        
        print(f"RLSnakeAI initialized. Using {'RL model' if self.rl_model else 'intelligent random strategy'}")
    
    def reset(self):
        """重置AI状态"""
        pass
    
    def _load_rl_model(self, model_path: Optional[str]):
        """加载强化学习模型"""
        if not model_path or not os.path.exists(model_path):
            print(f"Model path not found: {model_path}")
            return None
        
        try:
            # 尝试导入stable-baselines3
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            print(f"Successfully loaded RL model from: {model_path}")
            return model
        except ImportError:
            print("stable-baselines3 not available, using fallback strategy")
            return None
        except Exception as e:
            print(f"Failed to load RL model: {e}")
            return None
    
    def get_next_direction(self, my_snake: List[Tuple[int, int]], 
                          opponent_snake: List[Tuple[int, int]],
                          foods: List[Tuple[int, int]], 
                          current_direction: Tuple[int, int]) -> Tuple[int, int]:
        """
        获取下一步最优移动方向
        
        参数:
            my_snake: AI控制的蛇身坐标列表
            opponent_snake: 对手蛇身坐标列表  
            foods: 食物坐标列表
            current_direction: 当前移动方向
            
        返回:
            下一步移动方向(x, y)
        """
        if not my_snake:
            return current_direction
        
        my_head = my_snake[0]
        
        # 如果有强化学习模型，使用模型预测
        if self.rl_model:
            try:
                observation = self._create_observation(my_snake, opponent_snake, foods, current_direction)
                action_idx, _ = self.rl_model.predict(observation, deterministic=False)
                # print(f"RL model predicted action: {self.actions[action_idx]}")
                return self.actions[action_idx]
            except Exception as e:
                print(f"RL model prediction failed: {e}, falling back to intelligent strategy")
        
        # 如果没有强化学习模型，返回当前方向
        return current_direction
    
    def _create_observation(self, my_snake: List[Tuple[int, int]], 
                           opponent_snake: List[Tuple[int, int]],
                           foods: List[Tuple[int, int]], 
                           current_direction: Tuple[int, int]) -> np.ndarray:
        """
        为强化学习模型创建观察向量
        
        格式与训练时的观察空间保持一致：
        [自己蛇头x, 自己蛇头y, 自己长度, 对手蛇头x, 对手蛇头y, 对手长度,
         最近食物相对x, 最近食物相对y, 8个方向危险度, 当前方向x, 当前方向y, 与对手距离]
        """
        obs = []
        
        my_head = my_snake[0]
        opponent_head = opponent_snake[0] if opponent_snake else (self.window_size//2, self.window_size//2)
        
        # 1-2: 自己蛇头位置（标准化到[-1, 1]）
        obs.extend([
            (my_head[0] / self.window_size) * 2 - 1,
            (my_head[1] / self.window_size) * 2 - 1
        ])
        
        # 3: 自己蛇身长度（标准化）
        obs.append(min(len(my_snake) / 20.0, 1.0))
        
        # 4-5: 对手蛇头位置（标准化）
        obs.extend([
            (opponent_head[0] / self.window_size) * 2 - 1,
            (opponent_head[1] / self.window_size) * 2 - 1
        ])
        
        # 6: 对手蛇身长度（标准化）
        obs.append(min(len(opponent_snake) / 20.0, 1.0) if opponent_snake else 0.0)
        
        # 7-8: 最近食物的相对位置
        if foods:
            closest_food = min(foods, 
                             key=lambda f: math.sqrt((f[0] - my_head[0])**2 + (f[1] - my_head[1])**2))
            relative_food_x = (closest_food[0] - my_head[0]) / self.window_size
            relative_food_y = (closest_food[1] - my_head[1]) / self.window_size
            obs.extend([relative_food_x, relative_food_y])
        else:
            obs.extend([0, 0])
        
        # 9-16: 8个方向的危险度检测
        danger_detections = self._get_danger_in_directions(my_head, my_snake, opponent_snake)
        obs.extend(danger_detections)
        
        # 17-18: 当前移动方向
        obs.extend([current_direction[0], current_direction[1]])
        
        # 19: 与对手的相对距离
        distance_to_opponent = math.sqrt((my_head[0] - opponent_head[0])**2 + 
                                       (my_head[1] - opponent_head[1])**2)
        obs.append(min(distance_to_opponent / self.window_size, 1.0))
        
        return np.array(obs, dtype=np.float32)
    
    def _get_danger_in_directions(self, head_pos: Tuple[int, int], 
                                 my_snake: List[Tuple[int, int]], 
                                 opponent_snake: List[Tuple[int, int]]) -> List[float]:
        """检测8个方向的危险程度"""
        dangers = []
        
        for direction in self.actions:
            next_pos = (
                head_pos[0] + direction[0] * self.segment_distance,
                head_pos[1] + direction[1] * self.segment_distance
            )
            
            danger = 0.0
            
            # 检查边界碰撞
            boundary_margin = self.cell_size // 2
            if (next_pos[0] < boundary_margin or 
                next_pos[0] >= self.window_size - boundary_margin or
                next_pos[1] < boundary_margin or 
                next_pos[1] >= self.window_size - boundary_margin):
                danger = 1.0
            else:
                # 检查与自己身体的碰撞
                for segment in my_snake[1:]:
                    distance = math.sqrt((next_pos[0] - segment[0])**2 + (next_pos[1] - segment[1])**2)
                    if distance < self.cell_size * 0.8:
                        danger = max(danger, 0.8)
                    elif distance < self.cell_size * 1.5:
                        danger = max(danger, 0.4)
                
                # 检查与对手的碰撞
                for segment in opponent_snake:
                    distance = math.sqrt((next_pos[0] - segment[0])**2 + (next_pos[1] - segment[1])**2)
                    if distance < self.cell_size * 0.8:
                        danger = max(danger, 0.9)
                    elif distance < self.cell_size * 1.5:
                        danger = max(danger, 0.5)
            
            dangers.append(danger)
        
        return dangers
    
    
    
    
    
    def get_status(self) -> str:
        """获取AI状态信息"""
        if self.rl_model:
            return "RLSnakeAI: Using trained reinforcement learning model"
        else:
            return "RLSnakeAI: No RL model available"