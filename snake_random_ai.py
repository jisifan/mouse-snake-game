# -*- coding: utf-8 -*-
"""
随机贪吃蛇AI控制器
==================

一个具有智能安全检测的随机行为贪吃蛇AI，
会避免明显的危险但保持随机移动策略。

作者: Claude Code Assistant
依赖: 无（纯Python实现）
"""

import random
import math
from typing import List, Tuple

class RandomSnakeAI:
    """
    随机行为贪吃蛇AI控制器
    
    这个AI使用随机策略移动，但会进行基本的安全检测：
    - 避免撞墙
    - 避免撞到自己的身体
    - 避免撞到对手蛇身
    - 在多个安全方向中随机选择
    """
    
    def __init__(self, window_size: int, cell_size: int, segment_distance: int):
        """
        初始化随机AI控制器
        
        参数:
            window_size: 游戏窗口大小
            cell_size: 蛇身和食物的大小
            segment_distance: 蛇身节段间的距离
        """
        self.window_size = window_size
        self.cell_size = cell_size
        self.segment_distance = segment_distance
        
        # 八个移动方向：上下左右 + 对角线方向
        self.directions = [
            (0, -1), (0, 1), (-1, 0), (1, 0),  # 上、下、左、右
            (-1, -1), (1, -1), (-1, 1), (1, 1)  # 左上、右上、左下、右下
        ]
        
    def get_next_direction(self, my_snake: List[Tuple[int, int]], 
                          opponent_snake: List[Tuple[int, int]],
                          foods: List[Tuple[int, int]], 
                          current_direction: Tuple[int, int]) -> Tuple[int, int]:
        """
        获取下一步移动方向
        
        参数:
            my_snake: AI控制的蛇身坐标列表（索引0为蛇头）
            opponent_snake: 对手蛇身坐标列表
            foods: 食物坐标列表（未使用，但保持接口一致性）
            current_direction: 当前移动方向
            
        返回:
            tuple: 下一步移动方向(x, y)
        """
        if not my_snake:
            return current_direction
            
        my_head = my_snake[0]
        
        # 获取所有安全的移动方向
        safe_directions = []
        for direction in self.directions:
            # 计算按该方向移动后的新位置
            next_pos = (
                my_head[0] + direction[0] * self.segment_distance,
                my_head[1] + direction[1] * self.segment_distance
            )
            
            # 检查该位置是否安全
            if self._is_position_safe(next_pos, my_snake, opponent_snake):
                safe_directions.append(direction)
        
        # 如果有安全方向，随机选择一个
        if safe_directions:
            return random.choice(safe_directions)
        
        # 如果没有安全方向，返回当前方向或随机方向（紧急情况）
        return current_direction if current_direction != (0, 0) else random.choice(self.directions)
    
    def _is_position_safe(self, pos: Tuple[int, int], 
                         my_snake: List[Tuple[int, int]],
                         opponent_snake: List[Tuple[int, int]]) -> bool:
        """
        检查指定位置是否安全
        
        参数:
            pos: 要检查的位置坐标
            my_snake: 自己的蛇身
            opponent_snake: 对手蛇身
            
        返回:
            bool: 如果位置安全返回True，否则返回False
        """
        # ============= 边界碰撞检测 =============
        boundary_margin = self.cell_size // 2
        if (pos[0] < boundary_margin or 
            pos[0] >= self.window_size - boundary_margin or
            pos[1] < boundary_margin or 
            pos[1] >= self.window_size - boundary_margin):
            return False  # 会撞墙，不安全
            
        # ============= 自身碰撞检测 =============
        # 检查是否会撞到自己的身体（跳过蛇头）
        for segment in my_snake[1:]:
            distance = math.sqrt((pos[0] - segment[0])**2 + (pos[1] - segment[1])**2)
            if distance < self.cell_size * 0.8:
                return False  # 会撞到自己，不安全
                
        # ============= 对手碰撞检测 =============
        # 检查是否会撞到对手蛇身
        for segment in opponent_snake:
            distance = math.sqrt((pos[0] - segment[0])**2 + (pos[1] - segment[1])**2)
            if distance < self.cell_size * 0.8:
                return False  # 会撞到对手，不安全
                
        return True  # 位置安全
    
    def get_status(self) -> str:
        """
        获取AI状态信息（用于调试）
        
        返回:
            str: AI当前状态的描述
        """
        return "RandomSnakeAI: 使用随机但安全的移动策略"