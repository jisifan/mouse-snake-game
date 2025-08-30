# -*- coding: utf-8 -*-
"""
贪吃蛇强化学习环境 - 高效AI版本
================================

基于OpenAI Gym的贪吃蛇多智能体对抗环境，专为训练高效积极的AI设计：

核心特色：
- 递进式里程碑奖励：5食物(+200)、10食物(+300)、20食物(+500)
- 移动效率奖励：接近食物+5分，远离食物-3分
- 无效循环惩罚：原地打转-10分
- 竞争激励：领先优势奖励，追赶动机
- 大幅食物奖励：每个食物+50分直接奖励

设计理念：
1. 短期可达目标激励持续进步
2. 效率导向防止无意义移动
3. 竞争机制保持对抗性
4. 重奖励轻惩罚，积极强化学习

作者: Claude Code Assistant
依赖: gymnasium, numpy, pygame
"""

import gymnasium as gym
import numpy as np
import pygame
import math
import random
from typing import Tuple, List, Dict, Optional
from gymnasium import spaces

class SnakeRLEnv(gym.Env):
    """
    贪吃蛇强化学习环境
    
    支持两条蛇的对抗训练，每条蛇都作为独立的智能体，
    通过强化学习算法学习最优的移动策略。
    """
    
    def __init__(self, window_size: int = 600, cell_size: int = 20, segment_distance: int = 15):
        """
        初始化强化学习环境
        
        参数:
            window_size: 游戏窗口大小
            cell_size: 蛇身大小
            segment_distance: 蛇身节段距离
        """
        super().__init__()
        
        # ============= 环境基础参数 =============
        self.window_size = window_size
        self.cell_size = cell_size
        self.segment_distance = segment_distance
        self.max_steps = 1000  # 最大步数，防止无限循环
        
        # ============= 动作空间定义 =============
        # 8个方向的移动：上下左右 + 对角线
        self.action_space = spaces.Discrete(8)
        self.actions = [
            (0, -1), (0, 1), (-1, 0), (1, 0),  # 上、下、左、右
            (-1, -1), (1, -1), (-1, 1), (1, 1)  # 左上、右上、左下、右下
        ]
        
        # ============= 观察空间定义 =============
        # 状态向量：[自己蛇头x, 自己蛇头y, 自己长度, 对手蛇头x, 对手蛇头y, 对手长度,
        #          最近食物相对x, 最近食物相对y, 8个方向危险度, 当前方向x, 当前方向y]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(19,),  # 6 + 2 + 8 + 2 + 1 = 19维状态
            dtype=np.float32
        )
        
        # ============= 游戏状态变量 =============
        self.snake1_trail = []  # 蛇1的移动轨迹
        self.snake2_trail = []  # 蛇2的移动轨迹
        self.snake1_length = 1  # 蛇1长度
        self.snake2_length = 1  # 蛇2长度
        self.snake1 = []        # 蛇1身体位置
        self.snake2 = []        # 蛇2身体位置
        self.foods = []         # 食物列表
        self.current_step = 0   # 当前步数
        
        # ============= 得分和里程碑系统 =============
        self.snake1_score = 0   # 蛇1得分（基于吃到的食物数量）
        self.snake2_score = 0   # 蛇2得分
        
        # 递进式里程碑系统（更合理的短期目标）
        self.snake1_milestones = {'5': False, '10': False, '20': False}  # 里程碑达成状态
        self.snake2_milestones = {'5': False, '10': False, '20': False}
        
        # 效率追踪系统
        self.snake1_last_positions = []  # 记录最近的位置用于检测循环
        self.snake2_last_positions = []
        self.position_history_limit = 20  # 位置历史记录限制
        
        # 食物距离追踪（用于效率奖励）
        self.snake1_last_food_distance = None
        self.snake2_last_food_distance = None
        
        # 当前方向（用于状态表示）
        self.snake1_direction = (0, 0)
        self.snake2_direction = (0, 0)
        
        # 初始化pygame（用于可视化）
        pygame.init()
        self.screen = None
        self.clock = None
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        重置环境到初始状态
        
        返回:
            两条蛇的初始观察状态
        """
        # ============= 重置游戏状态 =============
        center_x = self.window_size // 2
        center_y = self.window_size // 2
        
        # 初始化两条蛇的位置（分开放置）
        self.snake1_trail = [(center_x - 100, center_y)]
        self.snake2_trail = [(center_x + 100, center_y)]
        self.snake1_length = 1
        self.snake2_length = 1
        self.snake1 = [(center_x - 100, center_y)]
        self.snake2 = [(center_x + 100, center_y)]
        
        # 重置方向
        self.snake1_direction = (0, 0)
        self.snake2_direction = (0, 0)
        
        # 生成初始食物
        self.foods = []
        self.generate_foods(5)
        
        # 重置步数和得分系统
        self.current_step = 0
        self.snake1_score = 0
        self.snake2_score = 0
        
        # 重置递进式里程碑
        self.snake1_milestones = {'5': False, '10': False, '20': False}
        self.snake2_milestones = {'5': False, '10': False, '20': False}
        
        # 重置效率追踪
        self.snake1_last_positions = []
        self.snake2_last_positions = []
        self.snake1_last_food_distance = None
        self.snake2_last_food_distance = None
        
        # 返回初始观察
        obs1 = self._get_observation(1)
        obs2 = self._get_observation(2)
        
        return obs1, obs2
    
    def step(self, actions: Tuple[int, int]) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                    Tuple[float, float], 
                                                    Tuple[bool, bool], 
                                                    Dict]:
        """
        执行一步动作
        
        参数:
            actions: 两条蛇的动作 (action1, action2)
            
        返回:
            observations: 新的观察状态
            rewards: 奖励
            dones: 是否结束
            info: 额外信息
        """
        action1, action2 = actions
        
        # ============= 执行动作 =============
        self.snake1_direction = self.actions[action1]
        self.snake2_direction = self.actions[action2]
        
        # 计算新的蛇头位置
        new_head1 = (
            self.snake1_trail[0][0] + self.snake1_direction[0] * self.segment_distance,
            self.snake1_trail[0][1] + self.snake1_direction[1] * self.segment_distance
        )
        new_head2 = (
            self.snake2_trail[0][0] + self.snake2_direction[0] * self.segment_distance,
            self.snake2_trail[0][1] + self.snake2_direction[1] * self.segment_distance
        )
        
        # ============= 检查碰撞和死亡 =============
        done1, done2, death_reason = self._check_collisions(new_head1, new_head2)
        
        # ============= 计算奖励 =============
        reward1, reward2 = self._calculate_rewards(new_head1, new_head2, done1, done2, death_reason)
        
        # ============= 更新游戏状态 =============
        if not done1:
            self._update_snake(1, new_head1)
        if not done2:
            self._update_snake(2, new_head2)
            
        # 检查食物碰撞
        self._check_food_collision(new_head1, new_head2)
        
        # 更新步数
        self.current_step += 1
        
        # 检查最大步数
        if self.current_step >= self.max_steps:
            done1 = done2 = True
        
        # ============= 返回结果 =============
        obs1 = self._get_observation(1)
        obs2 = self._get_observation(2)
        
        info = {
            'snake1_length': self.snake1_length,
            'snake2_length': self.snake2_length,
            'snake1_score': self.snake1_score,
            'snake2_score': self.snake2_score,
            'snake1_milestones': self.snake1_milestones.copy(),  # 递进式里程碑状态
            'snake2_milestones': self.snake2_milestones.copy(),  # 递进式里程碑状态
            'step': self.current_step,
            'death_reason': death_reason
        }
        
        return (obs1, obs2), (reward1, reward2), (done1, done2), info
    
    def _get_observation(self, snake_id: int) -> np.ndarray:
        """
        获取指定蛇的观察状态
        
        参数:
            snake_id: 蛇的ID (1 或 2)
            
        返回:
            标准化的观察向量
        """
        if snake_id == 1:
            my_snake = self.snake1
            my_trail = self.snake1_trail
            my_length = self.snake1_length
            my_direction = self.snake1_direction
            opponent_snake = self.snake2
            opponent_trail = self.snake2_trail
            opponent_length = self.snake2_length
        else:
            my_snake = self.snake2
            my_trail = self.snake2_trail
            my_length = self.snake2_length
            my_direction = self.snake2_direction
            opponent_snake = self.snake1
            opponent_trail = self.snake1_trail
            opponent_length = self.snake1_length
        
        if not my_trail or not opponent_trail:
            return np.zeros(19, dtype=np.float32)
            
        my_head = my_trail[0]
        opponent_head = opponent_trail[0]
        
        # ============= 构建观察向量 =============
        obs = []
        
        # 1-2: 自己蛇头位置（标准化到[-1, 1]）
        obs.extend([
            (my_head[0] / self.window_size) * 2 - 1,
            (my_head[1] / self.window_size) * 2 - 1
        ])
        
        # 3: 自己蛇身长度（标准化）
        obs.append(min(my_length / 20.0, 1.0))
        
        # 4-5: 对手蛇头位置（标准化）
        obs.extend([
            (opponent_head[0] / self.window_size) * 2 - 1,
            (opponent_head[1] / self.window_size) * 2 - 1
        ])
        
        # 6: 对手蛇身长度（标准化）
        obs.append(min(opponent_length / 20.0, 1.0))
        
        # 7-8: 最近食物的相对位置
        if self.foods:
            closest_food = min(self.foods, 
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
        obs.extend([my_direction[0], my_direction[1]])
        
        # 19: 与对手的相对距离
        distance_to_opponent = math.sqrt((my_head[0] - opponent_head[0])**2 + 
                                       (my_head[1] - opponent_head[1])**2)
        obs.append(min(distance_to_opponent / self.window_size, 1.0))
        
        return np.array(obs, dtype=np.float32)
    
    def _get_danger_in_directions(self, head_pos: Tuple[int, int], 
                                 my_snake: List[Tuple[int, int]], 
                                 opponent_snake: List[Tuple[int, int]]) -> List[float]:
        """
        检测8个方向的危险程度
        
        参数:
            head_pos: 蛇头位置
            my_snake: 自己的蛇身
            opponent_snake: 对手蛇身
            
        返回:
            8个方向的危险度列表（0-1，1表示最危险）
        """
        dangers = []
        
        for direction in self.actions:
            # 计算该方向前进一步的位置
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
                for segment in my_snake[1:]:  # 跳过蛇头
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
    
    def _check_collisions(self, new_head1: Tuple[int, int], new_head2: Tuple[int, int]) -> Tuple[bool, bool, str]:
        """
        检查碰撞和死亡条件
        
        参数:
            new_head1: 蛇1的新蛇头位置
            new_head2: 蛇2的新蛇头位置
            
        返回:
            (snake1_dead, snake2_dead, death_reason)
        """
        dead1, dead2 = False, False
        death_reason = ""
        
        boundary_left = self.cell_size // 2
        boundary_right = self.window_size - self.cell_size // 2
        boundary_top = self.cell_size // 2
        boundary_bottom = self.window_size - self.cell_size // 2
        
        # 检查边界碰撞
        if (new_head1[0] < boundary_left or new_head1[0] >= boundary_right or 
            new_head1[1] < boundary_top or new_head1[1] >= boundary_bottom):
            dead1 = True
            death_reason = "snake1_wall_collision"
            
        if (new_head2[0] < boundary_left or new_head2[0] >= boundary_right or 
            new_head2[1] < boundary_top or new_head2[1] >= boundary_bottom):
            dead2 = True
            death_reason = "snake2_wall_collision" if not dead1 else "both_wall_collision"
        
        # 检查自身碰撞
        if not dead1 and len(self.snake1) > 4:
            for segment in self.snake1[4:]:  # 跳过前几个节段
                distance = math.sqrt((new_head1[0] - segment[0])**2 + (new_head1[1] - segment[1])**2)
                if distance < self.cell_size * 0.7:
                    dead1 = True
                    death_reason = "snake1_self_collision"
                    break
        
        if not dead2 and len(self.snake2) > 4:
            for segment in self.snake2[4:]:
                distance = math.sqrt((new_head2[0] - segment[0])**2 + (new_head2[1] - segment[1])**2)
                if distance < self.cell_size * 0.7:
                    dead2 = True
                    death_reason = "snake2_self_collision" if not dead1 else "both_self_collision"
                    break
        
        # 检查蛇与蛇碰撞
        if not dead1:
            for segment in self.snake2:
                distance = math.sqrt((new_head1[0] - segment[0])**2 + (new_head1[1] - segment[1])**2)
                if distance < self.cell_size * 0.7:
                    dead1 = True
                    death_reason = "snake1_hit_snake2"
                    break
                    
        if not dead2:
            for segment in self.snake1:
                distance = math.sqrt((new_head2[0] - segment[0])**2 + (new_head2[1] - segment[1])**2)
                if distance < self.cell_size * 0.7:
                    dead2 = True
                    death_reason = "snake2_hit_snake1" if not dead1 else "mutual_collision"
                    break
        
        # 检查头部相撞
        head_distance = math.sqrt((new_head1[0] - new_head2[0])**2 + (new_head1[1] - new_head2[1])**2)
        if head_distance < self.cell_size * 0.7:
            dead1 = dead2 = True
            death_reason = "head_collision"
        
        return dead1, dead2, death_reason
    
    def _calculate_rewards(self, new_head1: Tuple[int, int], new_head2: Tuple[int, int], 
                          done1: bool, done2: bool, death_reason: str) -> Tuple[float, float]:
        """
        新的高效AI奖励系统
        
        设计理念：
        1. 递进式里程碑：5食物、10食物、20食物的阶梯奖励
        2. 效率导向：接近食物奖励，远离食物惩罚
        3. 无效移动惩罚：防止AI在原地打转
        4. 竞争激励：领先和追赶的动态奖励
        
        参数:
            new_head1: 蛇1新位置
            new_head2: 蛇2新位置
            done1: 蛇1是否死亡
            done2: 蛇2是否死亡
            death_reason: 死亡原因
            
        返回:
            (reward1, reward2)
        """
        reward1, reward2 = 0.0, 0.0
        
        # ============= 递进式里程碑奖励系统 =============
        reward1 += self._calculate_milestone_rewards(1)
        reward2 += self._calculate_milestone_rewards(2)
        
        # ============= 吃食物直接奖励 =============
        food_reward1, food_reward2 = self._calculate_food_rewards(new_head1, new_head2)
        reward1 += food_reward1
        reward2 += food_reward2
        
        # ============= 移动效率奖励/惩罚 =============
        if not done1:
            efficiency_reward1 = self._calculate_movement_efficiency(1, new_head1)
            reward1 += efficiency_reward1
            
        if not done2:
            efficiency_reward2 = self._calculate_movement_efficiency(2, new_head2)
            reward2 += efficiency_reward2
            
        # ============= 无效循环惩罚 =============
        if not done1:
            loop_penalty1 = self._check_movement_loops(1, new_head1)
            reward1 += loop_penalty1
            
        if not done2:
            loop_penalty2 = self._check_movement_loops(2, new_head2)
            reward2 += loop_penalty2
        
        # ============= 竞争优势奖励 =============
        competition_reward1, competition_reward2 = self._calculate_competition_rewards()
        reward1 += competition_reward1
        reward2 += competition_reward2
        
        # ============= 死亡惩罚 =============
        death_penalty1, death_penalty2 = self._calculate_death_penalties(done1, done2, death_reason)
        reward1 += death_penalty1
        reward2 += death_penalty2
        
        # ============= 基础存活奖励 =============
        if not done1:
            reward1 += 1.0  # 增加基础存活奖励，鼓励长期生存
        if not done2:
            reward2 += 1.0
        
        return reward1, reward2
    
    def _calculate_milestone_rewards(self, snake_id: int) -> float:
        """计算递进式里程碑奖励"""
        reward = 0.0
        score = self.snake1_score if snake_id == 1 else self.snake2_score
        milestones = self.snake1_milestones if snake_id == 1 else self.snake2_milestones
        
        # 检查5食物里程碑
        if score >= 5 and not milestones['5']:
            milestones['5'] = True
            reward += 200  # 5食物里程碑奖励
            print(f"Snake {snake_id} reached 5 food milestone! Score: {score}")
            
        # 检查10食物里程碑
        if score >= 10 and not milestones['10']:
            milestones['10'] = True
            reward += 300  # 10食物里程碑奖励
            print(f"Snake {snake_id} reached 10 food milestone! Score: {score}")
            
        # 检查20食物里程碑
        if score >= 20 and not milestones['20']:
            milestones['20'] = True
            reward += 500  # 20食物里程碑奖励
            print(f"Snake {snake_id} reached 20 food milestone! Score: {score}")
        
        return reward
    
    def _calculate_food_rewards(self, new_head1: Tuple[int, int], new_head2: Tuple[int, int]) -> Tuple[float, float]:
        """计算吃食物的直接奖励"""
        reward1, reward2 = 0.0, 0.0
        
        for food in self.foods:
            # 检查蛇1是否吃到食物
            if math.sqrt((new_head1[0] - food[0])**2 + (new_head1[1] - food[1])**2) < self.cell_size:
                reward1 += 50  # 大幅增加吃食物奖励
                
            # 检查蛇2是否吃到食物
            if math.sqrt((new_head2[0] - food[0])**2 + (new_head2[1] - food[1])**2) < self.cell_size:
                reward2 += 50  # 大幅增加吃食物奖励
                
        return reward1, reward2
    
    def _calculate_movement_efficiency(self, snake_id: int, new_head: Tuple[int, int]) -> float:
        """计算移动效率奖励/惩罚"""
        if not self.foods:
            return 0.0
            
        # 找到最近的食物
        closest_food = min(self.foods, key=lambda f: math.sqrt((f[0] - new_head[0])**2 + (f[1] - new_head[1])**2))
        current_distance = math.sqrt((closest_food[0] - new_head[0])**2 + (closest_food[1] - new_head[1])**2)
        
        # 获取上一次的距离
        if snake_id == 1:
            last_distance = self.snake1_last_food_distance
            self.snake1_last_food_distance = current_distance
        else:
            last_distance = self.snake2_last_food_distance
            self.snake2_last_food_distance = current_distance
            
        if last_distance is None:
            return 0.0  # 第一步没有比较基准
            
        # 计算效率奖励
        distance_change = last_distance - current_distance
        
        if distance_change > 0:
            # 接近食物，给予奖励
            return min(distance_change * 0.1, 5.0)  # 上限5分
        else:
            # 远离食物，给予惩罚
            return max(distance_change * 0.15, -3.0)  # 惩罚更重，下限-3分
    
    def _check_movement_loops(self, snake_id: int, new_head: Tuple[int, int]) -> float:
        """检查无效循环移动并给予惩罚"""
        if snake_id == 1:
            positions = self.snake1_last_positions
        else:
            positions = self.snake2_last_positions
            
        # 添加新位置
        positions.append(new_head)
        
        # 限制历史长度
        if len(positions) > self.position_history_limit:
            positions.pop(0)
            
        # 检查是否有重复的小范围移动
        if len(positions) >= 8:
            recent_positions = positions[-8:]
            
            # 计算位置的标准差，如果太小说明在原地打转
            x_coords = [pos[0] for pos in recent_positions]
            y_coords = [pos[1] for pos in recent_positions]
            
            x_std = np.std(x_coords) if len(set(x_coords)) > 1 else 0
            y_std = np.std(y_coords) if len(set(y_coords)) > 1 else 0
            
            # 如果移动范围太小，给予惩罚
            movement_range = x_std + y_std
            if movement_range < 20:  # 移动范围小于20像素认为是无效循环
                return -10.0  # 无效循环惩罚
                
        return 0.0
    
    def _calculate_competition_rewards(self) -> Tuple[float, float]:
        """计算竞争奖励"""
        score_diff = self.snake1_score - self.snake2_score
        
        if score_diff > 0:
            # 蛇1领先
            return min(score_diff * 2.0, 10.0), max(-score_diff * 1.0, -5.0)
        elif score_diff < 0:
            # 蛇2领先  
            return max(score_diff * 1.0, -5.0), min(abs(score_diff) * 2.0, 10.0)
        else:
            # 平局
            return 0.0, 0.0
    
    def _calculate_death_penalties(self, done1: bool, done2: bool, death_reason: str) -> Tuple[float, float]:
        """计算死亡惩罚"""
        penalty1, penalty2 = 0.0, 0.0
        
        if done1:
            if "snake1" in death_reason and "snake2" not in death_reason:
                penalty1 = -200  # 增加死亡惩罚
                penalty2 = +100  # 对手获胜奖励
            elif death_reason == "head_collision":
                penalty1 = -100  # 平局惩罚
                penalty2 = -100
        
        if done2:
            if "snake2" in death_reason and "snake1" not in death_reason:
                penalty2 = -200  # 增加死亡惩罚
                penalty1 = +100  # 对手获胜奖励
                
        return penalty1, penalty2
    
    def _update_snake(self, snake_id: int, new_head: Tuple[int, int]):
        """更新蛇的位置"""
        if snake_id == 1:
            self.snake1_trail.insert(0, new_head)
            max_trail_length = self.snake1_length * self.segment_distance + 100
            if len(self.snake1_trail) > max_trail_length:
                self.snake1_trail = self.snake1_trail[:max_trail_length]
            self._update_snake_body(1)
        else:
            self.snake2_trail.insert(0, new_head)
            max_trail_length = self.snake2_length * self.segment_distance + 100
            if len(self.snake2_trail) > max_trail_length:
                self.snake2_trail = self.snake2_trail[:max_trail_length]
            self._update_snake_body(2)
    
    def _update_snake_body(self, snake_id: int):
        """更新蛇身体位置"""
        if snake_id == 1:
            trail, length = self.snake1_trail, self.snake1_length
            self.snake1 = []
            for i in range(length):
                distance = (i * 0.8 + 0.2) * self.segment_distance if i > 0 else 0
                position = self._get_position_on_trail(distance, trail)
                self.snake1.append(position)
        else:
            trail, length = self.snake2_trail, self.snake2_length
            self.snake2 = []
            for i in range(length):
                distance = (i * 0.8 + 0.2) * self.segment_distance if i > 0 else 0
                position = self._get_position_on_trail(distance, trail)
                self.snake2.append(position)
    
    def _get_position_on_trail(self, distance: float, trail: List[Tuple[int, int]]) -> Tuple[int, int]:
        """在轨迹上根据距离找到位置"""
        if not trail or distance <= 0:
            return trail[0] if trail else (self.window_size//2, self.window_size//2)
        
        current_distance = 0
        for i in range(len(trail) - 1):
            point1 = trail[i]
            point2 = trail[i + 1]
            segment_distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
            
            if current_distance + segment_distance >= distance:
                remaining_distance = distance - current_distance
                ratio = remaining_distance / segment_distance if segment_distance > 0 else 0
                x = point1[0] + (point2[0] - point1[0]) * ratio
                y = point1[1] + (point2[1] - point1[1]) * ratio
                return (x, y)
            
            current_distance += segment_distance
        
        return trail[-1]
    
    def _check_food_collision(self, new_head1: Tuple[int, int], new_head2: Tuple[int, int]):
        """检查食物碰撞"""
        eaten_indices = []
        
        for i, food in enumerate(self.foods):
            # 检查蛇1
            if math.sqrt((new_head1[0] - food[0])**2 + (new_head1[1] - food[1])**2) < self.cell_size:
                eaten_indices.append(i)
                self.snake1_length += 2
                self.snake1_score += 1  # 每吃一个食物得1分（用于里程碑计算）
                continue
            
            # 检查蛇2
            if math.sqrt((new_head2[0] - food[0])**2 + (new_head2[1] - food[1])**2) < self.cell_size:
                eaten_indices.append(i)
                self.snake2_length += 2
                self.snake2_score += 1  # 每吃一个食物得1分（用于里程碑计算）
        
        # 删除被吃的食物
        for i in sorted(eaten_indices, reverse=True):
            self.foods.pop(i)
        
        # 生成新食物
        if eaten_indices:
            self.generate_foods(len(eaten_indices))
    
    def generate_foods(self, count: int):
        """生成指定数量的食物"""
        for _ in range(count):
            if len(self.foods) >= 10:
                break
            
            # 简化食物生成逻辑
            for attempt in range(50):
                food_x = random.randint(self.cell_size, self.window_size - self.cell_size)
                food_y = random.randint(self.cell_size, self.window_size - self.cell_size)
                food = (food_x, food_y)
                
                # 简单的碰撞检测
                too_close = False
                for snake in [self.snake1, self.snake2]:
                    for segment in snake:
                        distance = math.sqrt((food[0] - segment[0])**2 + (food[1] - segment[1])**2)
                        if distance < self.cell_size * 1.5:
                            too_close = True
                            break
                    if too_close:
                        break
                
                if not too_close:
                    for existing_food in self.foods:
                        distance = math.sqrt((food[0] - existing_food[0])**2 + (food[1] - existing_food[1])**2)
                        if distance < self.cell_size * 2:
                            too_close = True
                            break
                
                if not too_close:
                    self.foods.append(food)
                    break
    
    def render(self, mode='human'):
        """渲染环境（可选）"""
        if mode == 'human':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("Snake RL Training")
                self.clock = pygame.time.Clock()
            
            # 清空屏幕
            self.screen.fill((0, 0, 0))
            
            # 绘制蛇1（绿色）
            for i, segment in enumerate(self.snake1):
                color = (0, 150, 0) if i == 0 else (0, 255, 0)
                pygame.draw.circle(self.screen, color, 
                                 (int(segment[0]), int(segment[1])), 
                                 self.cell_size // 2 - 1)
            
            # 绘制蛇2（蓝色）
            for i, segment in enumerate(self.snake2):
                color = (0, 50, 150) if i == 0 else (0, 100, 255)
                pygame.draw.circle(self.screen, color, 
                                 (int(segment[0]), int(segment[1])), 
                                 self.cell_size // 2 - 1)
            
            # 绘制食物
            for food in self.foods:
                pygame.draw.circle(self.screen, (255, 0, 0), 
                                 (int(food[0]), int(food[1])), 
                                 self.cell_size // 2 - 1)
            
            pygame.display.flip()
            self.clock.tick(60)
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None