"""
智能贪吃蛇AI控制器
====================

一个基于A*寻路算法和安全性评估的贪吃蛇AI系统。
能够自动寻找食物、避免碰撞、预测危险。

主要特性：
- A*寻路算法找到最优路径
- 多层安全性检查避免碰撞
- 启发式决策优化生存策略
- 实时环境感知和路径规划

作者: Claude Code Assistant
依赖: 无（纯Python实现）
"""

import math
import heapq
from typing import List, Tuple, Optional, Set

class SnakeAI:
    """
    智能贪吃蛇AI控制器
    
    使用A*算法和启发式评估来控制蛇的移动，实现：
    - 自动寻找最近的安全食物
    - 避免与自己身体碰撞
    - 避免与对手蛇碰撞  
    - 避免撞墙
    - 当无法安全到达食物时保持安全移动
    """
    
    def __init__(self, window_size: int, cell_size: int, segment_distance: int):
        """
        初始化AI控制器
        
        参数:
            window_size: 游戏窗口大小
            cell_size: 细胞大小
            segment_distance: 蛇身节段间距
        """
        self.window_size = window_size
        self.cell_size = cell_size
        self.segment_distance = segment_distance
        
        # 八个移动方向：上下左右 + 对角线方向
        self.directions = [
            (0, -1), (0, 1), (-1, 0), (1, 0),  # 上、下、左、右
            (-1, -1), (1, -1), (-1, 1), (1, 1)  # 左上、右上、左下、右下
        ]
        
        # 方向名称映射（用于调试）
        self.direction_names = {
            (0, -1): "UP",
            (0, 1): "DOWN", 
            (-1, 0): "LEFT",
            (1, 0): "RIGHT",
            (-1, -1): "LEFT_UP",
            (1, -1): "RIGHT_UP", 
            (-1, 1): "LEFT_DOWN",
            (1, 1): "RIGHT_DOWN",
            (0, 0): "STOP"
        }
        
    def get_next_direction(self, my_snake: List[Tuple[int, int]], 
                          opponent_snake: List[Tuple[int, int]],
                          foods: List[Tuple[int, int]], 
                          current_direction: Tuple[int, int]) -> Tuple[int, int]:
        """
        AI决策核心方法：计算下一步最优移动方向（优化食物竞争策略）
        
        参数:
            my_snake: AI控制的蛇身坐标列表（索引0为蛇头）
            opponent_snake: 对手蛇身坐标列表
            foods: 食物坐标列表
            current_direction: 当前移动方向
            
        返回:
            tuple: 下一步移动方向(x, y)
        """
        if not my_snake or not foods:
            return current_direction
            
        my_head = my_snake[0]
        opponent_head = opponent_snake[0] if opponent_snake else None
        
        # 第一步：分析所有食物的竞争优势
        food_priorities = []
        
        for food in foods:
            # 计算到食物的距离
            my_distance = self._heuristic_distance(my_head, food)
            
            # 计算竞争优势分数
            priority_score = self._calculate_food_priority(
                food, my_head, opponent_head, my_distance
            )
            
            food_priorities.append((food, priority_score, my_distance))
        
        # 按优先级排序（优先级高的在前）
        food_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # 第二步：尝试寻找到高优先级食物的安全路径
        best_direction = None
        best_path_length = float('inf')
        
        for food, priority, my_distance in food_priorities:
            # 使用A*算法寻找到该食物的路径
            path = self._find_path_to_food(my_head, food, my_snake, opponent_snake)
            
            if path and len(path) > 1:
                # 找到了安全路径
                next_pos = path[1]  # 路径中的下一个位置
                direction = (next_pos[0] - my_head[0], next_pos[1] - my_head[1])
                
                # 标准化方向向量
                direction = self._normalize_direction(direction)
                
                # 评估这个方向的安全性
                if self._is_direction_safe(my_head, direction, my_snake, opponent_snake):
                    # 优先选择高优先级的食物，其次考虑路径长度
                    path_score = len(path) - priority * 10  # 优先级权重
                    if path_score < best_path_length:
                        best_direction = direction
                        best_path_length = path_score
        
        # 如果找到了到食物的安全路径，使用它
        if best_direction is not None:
            return best_direction
            
        # 如果无法安全到达任何食物，执行安全移动策略
        return self._get_safe_direction(my_head, my_snake, opponent_snake, current_direction)
    
    def _calculate_food_priority(self, food: Tuple[int, int], 
                               my_head: Tuple[int, int],
                               opponent_head: Optional[Tuple[int, int]],
                               my_distance: float) -> float:
        """
        计算食物的竞争优先级分数
        
        参数:
            food: 食物位置
            my_head: 我的蛇头位置
            opponent_head: 对手蛇头位置
            my_distance: 我到食物的距离
            
        返回:
            优先级分数（越高越优先）
        """
        priority = 0.0
        
        if opponent_head:
            opponent_distance = self._heuristic_distance(opponent_head, food)
            
            # 如果我距离更近，获得竞争优势
            if my_distance < opponent_distance:
                # 距离优势越大，优先级越高
                distance_advantage = opponent_distance - my_distance
                priority += distance_advantage * 2.0
            else:
                # 如果对手更近，降低优先级
                distance_disadvantage = my_distance - opponent_distance
                priority -= distance_disadvantage * 1.5
            
            # 如果对手距离食物很近（可能正在争夺），提高紧急程度
            if opponent_distance < 50:  # 50像素内认为是紧急竞争
                if my_distance < opponent_distance:
                    priority += 50  # 我有优势，大幅提升优先级
                else:
                    priority -= 30  # 我处于劣势，但仍要尝试
        
        # 基础优先级：越近的食物优先级越高
        base_priority = max(0, 200 - my_distance)
        priority += base_priority
        
        return priority
    
    def _find_path_to_food(self, start: Tuple[int, int], goal: Tuple[int, int],
                          my_snake: List[Tuple[int, int]], 
                          opponent_snake: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """
        使用A*算法寻找从起点到食物的最优路径
        
        参数:
            start: 起始位置（蛇头位置）
            goal: 目标位置（食物位置）
            my_snake: 自己的蛇身
            opponent_snake: 对手蛇身
            
        返回:
            路径坐标列表，如果无法到达返回None
        """
        # A*算法实现
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic_distance(start, goal)}
        
        visited = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # 找到目标，重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # 反转得到正确顺序
                
            visited.add(current)
            
            # 探索八个方向
            for direction in self.directions:
                neighbor = (
                    current[0] + direction[0] * self.segment_distance,
                    current[1] + direction[1] * self.segment_distance
                )
                
                # 检查邻居位置是否合法和安全
                if (neighbor in visited or 
                    not self._is_position_safe(neighbor, my_snake, opponent_snake)):
                    continue
                
                # 计算移动成本（对角线移动成本更高）
                if direction[0] != 0 and direction[1] != 0:
                    # 对角线移动：距离为 segment_distance * √2
                    move_cost = int(self.segment_distance * 1.414)
                else:
                    # 直线移动：距离为 segment_distance
                    move_cost = self.segment_distance
                
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic_distance(neighbor, goal)
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # 无法找到路径
    
    def _heuristic_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        启发式距离函数（曼哈顿距离）
        
        参数:
            pos1: 位置1
            pos2: 位置2
            
        返回:
            两点间的曼哈顿距离
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _is_position_safe(self, pos: Tuple[int, int], 
                         my_snake: List[Tuple[int, int]],
                         opponent_snake: List[Tuple[int, int]]) -> bool:
        """
        检查位置是否安全（不会碰撞）
        
        参数:
            pos: 要检查的位置
            my_snake: 自己的蛇身
            opponent_snake: 对手蛇身
            
        返回:
            如果位置安全返回True，否则返回False
        """
        # 检查边界
        boundary_margin = self.cell_size // 2
        if (pos[0] < boundary_margin or 
            pos[0] >= self.window_size - boundary_margin or
            pos[1] < boundary_margin or 
            pos[1] >= self.window_size - boundary_margin):
            return False
            
        # 检查与自己身体的碰撞（跳过蛇头）
        for segment in my_snake[1:]:
            if self._positions_collide(pos, segment):
                return False
                
        # 检查与对手的碰撞
        for segment in opponent_snake:
            if self._positions_collide(pos, segment):
                return False
                
        return True
    
    def _positions_collide(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """
        检查两个位置是否碰撞
        
        参数:
            pos1: 位置1
            pos2: 位置2
            
        返回:
            如果碰撞返回True
        """
        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return distance < self.cell_size * 0.8
    
    def _normalize_direction(self, direction: Tuple[int, int]) -> Tuple[int, int]:
        """
        标准化方向向量为单位方向（支持8方向）
        
        参数:
            direction: 原始方向向量
            
        返回:
            标准化后的方向向量
        """
        if direction[0] == 0 and direction[1] == 0:
            return (0, 0)
            
        # 获取方向的符号
        dx = 1 if direction[0] > 0 else (-1 if direction[0] < 0 else 0)
        dy = 1 if direction[1] > 0 else (-1 if direction[1] < 0 else 0)
        
        return (dx, dy)
    
    def _is_direction_safe(self, head_pos: Tuple[int, int], direction: Tuple[int, int],
                          my_snake: List[Tuple[int, int]], 
                          opponent_snake: List[Tuple[int, int]]) -> bool:
        """
        检查特定方向是否安全
        
        参数:
            head_pos: 蛇头当前位置
            direction: 要检查的方向
            my_snake: 自己的蛇身
            opponent_snake: 对手蛇身
            
        返回:
            方向是否安全
        """
        next_pos = (
            head_pos[0] + direction[0] * self.segment_distance,
            head_pos[1] + direction[1] * self.segment_distance
        )
        
        return self._is_position_safe(next_pos, my_snake, opponent_snake)
    
    def _get_safe_direction(self, head_pos: Tuple[int, int],
                           my_snake: List[Tuple[int, int]], 
                           opponent_snake: List[Tuple[int, int]],
                           current_direction: Tuple[int, int]) -> Tuple[int, int]:
        """
        当无法到达食物时，获取一个安全的移动方向
        
        参数:
            head_pos: 蛇头位置
            my_snake: 自己的蛇身
            opponent_snake: 对手蛇身
            current_direction: 当前移动方向
            
        返回:
            安全的移动方向
        """
        # 优先级：继续当前方向 > 其他安全方向 > 停止
        
        # 首先尝试继续当前方向
        if (current_direction != (0, 0) and 
            self._is_direction_safe(head_pos, current_direction, my_snake, opponent_snake)):
            return current_direction
            
        # 寻找其他安全方向，优先选择远离边界的方向
        safe_directions = []
        for direction in self.directions:
            if self._is_direction_safe(head_pos, direction, my_snake, opponent_snake):
                # 计算这个方向的"安全度"（距离边界的距离）
                next_pos = (
                    head_pos[0] + direction[0] * self.segment_distance,
                    head_pos[1] + direction[1] * self.segment_distance
                )
                safety_score = self._calculate_position_safety(next_pos, my_snake, opponent_snake)
                safe_directions.append((direction, safety_score))
        
        if safe_directions:
            # 选择安全度最高的方向
            safe_directions.sort(key=lambda x: x[1], reverse=True)
            return safe_directions[0][0]
            
        # 如果没有安全方向，保持当前方向或停止
        return current_direction if current_direction != (0, 0) else (0, 1)  # 默认向下
    
    def _calculate_position_safety(self, pos: Tuple[int, int],
                                  my_snake: List[Tuple[int, int]], 
                                  opponent_snake: List[Tuple[int, int]]) -> float:
        """
        计算位置的安全度分数
        
        参数:
            pos: 位置坐标
            my_snake: 自己的蛇身
            opponent_snake: 对手蛇身
            
        返回:
            安全度分数（越高越安全）
        """
        safety_score = 0.0
        
        # 距离边界的距离（越远越安全）
        boundary_margin = self.cell_size // 2
        dist_to_left = pos[0] - boundary_margin
        dist_to_right = self.window_size - boundary_margin - pos[0]
        dist_to_top = pos[1] - boundary_margin
        dist_to_bottom = self.window_size - boundary_margin - pos[1]
        
        min_boundary_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        safety_score += min_boundary_dist
        
        # 距离自己身体的最小距离（越远越安全）
        min_self_dist = float('inf')
        for segment in my_snake[1:]:  # 跳过蛇头
            dist = math.sqrt((pos[0] - segment[0])**2 + (pos[1] - segment[1])**2)
            min_self_dist = min(min_self_dist, dist)
        
        if min_self_dist != float('inf'):
            safety_score += min_self_dist * 0.5
            
        # 距离对手的最小距离（越远越安全）
        min_opponent_dist = float('inf')
        for segment in opponent_snake:
            dist = math.sqrt((pos[0] - segment[0])**2 + (pos[1] - segment[1])**2)
            min_opponent_dist = min(min_opponent_dist, dist)
        
        if min_opponent_dist != float('inf'):
            safety_score += min_opponent_dist * 0.3
            
        return safety_score
    
    def get_status(self) -> str:
        """
        获取AI状态信息（用于调试）
        
        返回:
            AI当前状态的字符串描述
        """
        return "SnakeAI: Ready for intelligent control"