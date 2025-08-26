"""
鼠标控制贪吃蛇游戏
==================

一个创新的贪吃蛇游戏实现，具有以下特色功能：
- 完全鼠标控制，蛇头精确跟随鼠标位置
- 轨迹跟随系统，蛇身沿着蛇头历史路径移动
- 智能多食物分布，维持3-10个食物的正态分布
- 蛇头增长机制，从头部而非尾部延长

作者: Claude Code Assistant
依赖: pygame
"""

import pygame
import random
import sys
import math
from snake_ai import SnakeAI

# 初始化Pygame
pygame.init()

# ============= 游戏常量定义 =============
WINDOW_SIZE = 600           # 游戏窗口大小（像素）
CELL_SIZE = 20             # 蛇身和食物的基础大小（像素）
SEGMENT_DISTANCE = 15      # 蛇身节段间的固定距离（像素）
SNAKE2_MOVE_INTERVAL = 80  # AI初始移动间隔（毫秒）- 提高决策频率
GRID_WIDTH = WINDOW_SIZE // CELL_SIZE    # 网格宽度（用于边界计算）
GRID_HEIGHT = WINDOW_SIZE // CELL_SIZE   # 网格高度（用于边界计算）

# ============= 颜色常量定义 =============
BLACK = (0, 0, 0)          # 背景色：黑色
GREEN = (0, 255, 0)        # 玩家1蛇身色：亮绿色
RED = (255, 0, 0)          # 食物色：红色
WHITE = (255, 255, 255)    # 文字色：白色
DARK_GREEN = (0, 150, 0)   # 玩家1蛇头色：暗绿色
BLUE = (0, 100, 255)       # 玩家2蛇身色：亮蓝色
DARK_BLUE = (0, 50, 150)   # 玩家2蛇头色：暗蓝色

class SnakeGame:
    """
    贪吃蛇游戏主类
    
    这个类实现了完整的贪吃蛇游戏逻辑，包括：
    - 基于轨迹跟随的蛇身移动系统
    - 智能多食物分布管理
    - 鼠标控制和碰撞检测
    - 游戏状态管理和渲染
    
    主要特色：
    - 蛇头完全跟随鼠标位置，无速度限制
    - 蛇身精确跟随蛇头的历史移动轨迹
    - 多个食物同时存在，智能维持正态分布
    - 从蛇头延长而非蛇尾延长的增长机制
    """
    
    def __init__(self):
        """
        初始化游戏
        
        设置游戏窗口、字体、鼠标设置等基础配置，
        并调用reset_game()初始化游戏状态。
        """
        # 创建游戏窗口
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Snake Game")
        pygame.mouse.set_visible(False)  # 隐藏鼠标指针以获得更好的游戏体验
        
        # 初始化时钟控制器（控制帧率）
        self.clock = pygame.time.Clock()
        
        # 初始化AI控制器
        self.ai_controller = SnakeAI(WINDOW_SIZE, CELL_SIZE, SEGMENT_DISTANCE)
        
        # 尝试加载中文字体，失败则使用默认字体
        try:
            self.font = pygame.font.Font("C:/Windows/Fonts/simsun.ttc", 36)
        except:
            self.font = pygame.font.Font(None, 36)
            
        # 初始化游戏状态
        self.reset_game()
        
    def reset_game(self):
        """
        重置游戏状态
        
        将游戏重置到初始状态，包括：
        - 重置两条蛇的位置和长度
        - 清空移动轨迹
        - 重新生成食物
        - 重置双方分数和游戏状态标志
        """
        # 计算屏幕中心位置和初始位置
        center_x = WINDOW_SIZE // 2
        center_y = WINDOW_SIZE // 2
        
        # ============= 玩家1蛇身系统初始化（鼠标控制，轨迹跟随）=============
        # 轨迹跟随系统：记录蛇头的完整移动历史
        self.snake1_trail = [(center_x - 100, center_y)]  # 玩家1蛇头移动轨迹列表
        self.snake1_length = 1                            # 玩家1蛇的长度（节数）
        self.snake1 = [(center_x - 100, center_y)]        # 玩家1蛇身各节段的位置
        
        # ============= 玩家2蛇身系统初始化（键盘控制，方向移动）=============
        self.snake2_trail = [(center_x + 100, center_y)]  # 玩家2蛇头移动轨迹列表
        self.snake2_length = 1                            # 玩家2蛇的长度（节数）
        self.snake2 = [(center_x + 100, center_y)]        # 玩家2蛇身各节段的位置
        self.snake2_direction = (0, 0)                    # 玩家2初始方向：静止，等待按键
        self.snake2_pos = (center_x + 100, center_y)      # 玩家2当前位置
        self.snake2_last_move_time = 0                    # 玩家2上次移动时间
        
        # ============= 控制和输入系统 =============
        self.mouse_pos = (center_x - 100, center_y)       # 玩家1鼠标位置
        
        # ============= 食物系统初始化 =============
        self.foods = []                            # 食物列表（支持多个食物同时存在）
        self.generate_foods(5)                     # 初始生成5个食物
        
        # ============= 游戏状态变量 =============
        self.score1 = 0                           # 玩家1分数
        self.score2 = 0                           # 玩家2分数
        self.game_over = False                     # 游戏结束标志
        self.game_started = False                  # 游戏开始标志
        self.winner = None                         # 胜利者（1或2，None表示平局）
        self.pending_growth = 0                    # 待增长的身体节段数（暂未使用）
        
        # ============= AI动态速度系统 =============
        self.game_start_time = 0                   # 游戏开始时间（毫秒）
        self.current_ai_interval = SNAKE2_MOVE_INTERVAL  # 当前AI移动间隔
        self.speed_increase_rate = 0.98            # 速度递增率（每次递减2%间隔）
        self.min_ai_interval = 30                  # 最小AI移动间隔（毫秒）
        
    def generate_food(self):
        """
        生成单个食物
        
        在游戏区域内随机生成一个食物，确保：
        - 不与两条蛇身重叠（距离至少1.5倍CELL_SIZE）
        - 不与现有食物过近（距离至少2倍CELL_SIZE）
        - 在边界内的安全区域
        
        Returns:
            tuple: 食物的(x, y)坐标位置
            
        Note:
            最多尝试100次找到合适位置，如果都失败则返回随机位置
        """
        max_attempts = 100  # 最大尝试次数，防止无限循环
        
        for attempt in range(max_attempts):
            # 在安全边界内随机生成位置（避免食物贴边界）
            food_x = random.randint(CELL_SIZE, WINDOW_SIZE - CELL_SIZE)
            food_y = random.randint(CELL_SIZE, WINDOW_SIZE - CELL_SIZE)
            food = (food_x, food_y)
            
            # 检查与两条蛇身的距离冲突
            too_close_to_snake = False
            # 检查玩家1的蛇
            for segment in self.snake1:
                distance = math.sqrt((food[0] - segment[0])**2 + (food[1] - segment[1])**2)
                if distance < CELL_SIZE * 1.5:  # 与蛇身保持1.5倍安全距离
                    too_close_to_snake = True
                    break
            # 检查玩家2的蛇
            if not too_close_to_snake:
                for segment in self.snake2:
                    distance = math.sqrt((food[0] - segment[0])**2 + (food[1] - segment[1])**2)
                    if distance < CELL_SIZE * 1.5:  # 与蛇身保持1.5倍安全距离
                        too_close_to_snake = True
                        break
            
            # 检查与现有食物的距离冲突
            too_close_to_food = False
            for existing_food in self.foods:
                distance = math.sqrt((food[0] - existing_food[0])**2 + (food[1] - existing_food[1])**2)
                if distance < CELL_SIZE * 2:  # 食物间保持2倍安全距离，防止聚集
                    too_close_to_food = True
                    break
            
            # 如果位置合适，返回该位置
            if not too_close_to_snake and not too_close_to_food:
                return food
        
        # 如果尝试100次都没找到合适位置，返回一个随机位置（兜底策略）
        return (random.randint(CELL_SIZE, WINDOW_SIZE - CELL_SIZE), 
                random.randint(CELL_SIZE, WINDOW_SIZE - CELL_SIZE))
    
    def generate_foods(self, count):
        """
        批量生成指定数量的食物
        
        这是一个辅助方法，用于一次性生成多个食物。会调用generate_food()方法
        来确保每个生成的食物都符合位置要求和冲突检测规则。
        
        参数:
            count (int): 需要生成的食物数量
            
        注意:
            - 最多只能生成到总数10个食物的上限
            - 每个食物的生成都会经过完整的冲突检测
            - 如果无法找到合适位置会使用兜底策略
        """
        for _ in range(count):
            # 检查食物总数上限：不超过10个食物
            if len(self.foods) >= 10:  
                break  # 达到上限，停止生成更多食物
                
            # 生成一个新食物并添加到食物列表
            new_food = self.generate_food()
            self.foods.append(new_food)
    
    def manage_food_count(self):
        """
        管理食物数量：智能生成以达到正态分布
        
        这个方法实现了一个复杂的加权概率系统，通过不同的权重分布来控制食物生成：
        - 当食物数量偏离理想范围(5-7个)时，权重会向理想范围倾斜
        - 使用分段函数根据当前食物数量调整生成概率
        - 通过权重调整实现食物数量的自然回归和正态分布效果
        
        算法核心：
        1. 根据当前食物数量选择对应的权重分布策略
        2. 使用加权随机选择确定新生成的食物数量(0-3个)
        3. 执行边界检查确保最终数量在合理范围内(3-10个)
        """
        current_count = len(self.foods)
        
        # ============= 加权概率分布策略 =============
        # 根据当前食物数量动态调整生成权重，实现智能的正态分布控制
        # 权重数组格式：[生成0个, 生成1个, 生成2个, 生成3个]的概率权重
        
        if current_count <= 2:
            # 食物严重不足：强烈倾向生成更多食物
            # 策略：几乎不选择0个，大量倾向2-3个
            weights = [5, 15, 40, 40]  # 80%概率生成2-3个
        elif current_count == 3:
            # 食物偏少：倾向生成更多，但不如<=2时强烈
            # 策略：减少生成0个的概率，增加生成2-3个的概率
            weights = [10, 20, 35, 35]  # 70%概率生成2-3个
        elif current_count == 4:
            # 食物稍少：平衡策略，允许适度波动
            # 策略：相对平衡的分布，但仍略偏向生成更多
            weights = [25, 30, 30, 15]  # 45%概率生成2-3个
        elif current_count == 5:
            # 接近理想数量：略偏向减少，防止数量继续增长
            # 策略：增加生成0-1个的概率，减少生成3个的概率
            weights = [30, 35, 25, 10]  # 65%概率生成0-1个
        elif current_count == 6:
            # 理想数量：完全平衡的分布策略
            # 策略：四种选择等概率，让数量自然波动
            weights = [25, 25, 25, 25]  # 各选择等概率
        elif current_count == 7:
            # 理想上限：开始倾向减少食物数量
            # 策略：强烈偏向生成0-1个，避免数量继续增长
            weights = [45, 35, 15, 5]   # 80%概率生成0-1个
        elif current_count == 8:
            # 数量偏多：非常强烈倾向于不生成新食物
            # 策略：大幅增加生成0个的概率，严格控制新增
            weights = [60, 30, 8, 2]    # 90%概率生成0-1个
        elif current_count >= 9:
            # 数量过多：极其强烈倾向于不生成，接近上限
            # 策略：几乎只生成0个，偶尔生成1个，绝不生成3个
            weights = [80, 18, 2, 0]    # 98%概率生成0-1个
        
        # ============= 执行加权随机选择 =============
        # 使用Python的random.choices进行加权随机选择
        # choices参数：可选值列表 [0, 1, 2, 3]
        # weights参数：对应的权重列表
        choices = [0, 1, 2, 3]
        new_foods = random.choices(choices, weights=weights)[0]
        
        # ============= 执行食物生成 =============
        # 调用generate_foods方法生成指定数量的新食物
        self.generate_foods(new_foods)
        
        # ============= 边界安全检查 =============
        # 确保最终食物数量严格控制在3-10个范围内
        # 这是系统的最后一道防线，防止极端情况
        final_count = len(self.foods)
        if final_count < 3:
            # 数量低于下限：强制补充到最少3个
            needed = 3 - final_count
            self.generate_foods(needed)
        elif final_count > 10:
            # 数量超过上限：随机删除多余食物
            excess = final_count - 10
            for _ in range(excess):
                if self.foods:  # 防御性检查：确保列表不为空
                    # 随机选择一个食物进行删除，避免固定删除模式
                    random_index = random.randint(0, len(self.foods) - 1)
                    self.foods.pop(random_index)
    
    def get_position_on_trail(self, distance, snake_trail):
        """
        根据指定距离在蛇头移动轨迹上找到精确位置
        
        这是轨迹跟随系统的核心算法，实现蛇身节段在轨迹上的精确定位：
        
        算法原理：
        1. 轨迹由一系列连续的点组成，形成蛇头的历史移动路径
        2. 通过累积线段长度找到目标距离所在的具体线段
        3. 使用线性插值在该线段内计算精确位置坐标
        4. 确保每个蛇身节段都能在轨迹上找到准确位置
        
        数学模型：
        - 使用欧几里得距离公式计算线段长度: √[(x2-x1)² + (y2-y1)²]
        - 使用线性插值公式计算位置: P = P1 + t(P2-P1), 其中t为插值比率
        
        参数:
            distance (float): 从轨迹起点(蛇头位置)开始的距离
            snake_trail (list): 蛇头移动轨迹列表
            
        返回:
            tuple: 对应距离处的(x, y)坐标位置
            
        边界处理:
        - 距离为0或负数时返回蛇头位置
        - 距离超出轨迹总长度时返回轨迹末端位置
        """
        # ============= 边界条件检查 =============
        if not snake_trail or distance <= 0:
            # 轨迹为空或距离无效时，返回蛇头位置或默认中心位置
            return snake_trail[0] if snake_trail else (WINDOW_SIZE//2, WINDOW_SIZE//2)
        
        # ============= 轨迹遍历和距离累积 =============
        current_distance = 0  # 当前已累积的距离
        
        # 遍历轨迹中的每个线段(相邻两点之间的连线)
        for i in range(len(snake_trail) - 1):
            point1 = snake_trail[i]      # 线段起点(距离蛇头更近)
            point2 = snake_trail[i + 1]  # 线段终点(距离蛇头更远)
            
            # ============= 计算线段长度 =============
            # 使用欧几里得距离公式计算两点间直线距离
            segment_distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
            
            # ============= 检查目标距离是否在当前线段内 =============
            if current_distance + segment_distance >= distance:
                # 目标位置确定在当前线段上，开始精确定位
                
                # 计算在当前线段内还需要的距离
                remaining_distance = distance - current_distance
                
                # 计算插值比率：在线段上的相对位置(0.0到1.0)
                # ratio = 0.0 表示在point1位置，ratio = 1.0 表示在point2位置
                ratio = remaining_distance / segment_distance if segment_distance > 0 else 0
                
                # ============= 线性插值计算精确位置 =============
                # 使用线性插值公式：P = P1 + t(P2 - P1)
                # 其中t为插值参数(ratio)，P1和P2为线段端点
                x = point1[0] + (point2[0] - point1[0]) * ratio
                y = point1[1] + (point2[1] - point1[1]) * ratio
                return (x, y)
            
            # 当前线段不包含目标距离，累积距离继续查找下一个线段
            current_distance += segment_distance
        
        # ============= 处理超出轨迹长度的情况 =============
        # 如果目标距离超过整个轨迹的总长度，返回轨迹的最末端位置
        # 这确保了即使轨迹很短也能为每个蛇身节段找到合理位置
        return snake_trail[-1]
    
    def update_snake_body(self):
        """
        根据轨迹和蛇长度更新玩家1蛇身各节段位置 - 实现蛇头增长机制
        """
        # ============= 重置蛇身位置列表 =============
        # 清空当前蛇身，准备重新计算所有节段位置
        self.snake1 = []
        
        # ============= 逐节段计算位置 =============
        # 遍历蛇的每个节段，从蛇头(索引0)到蛇尾
        for i in range(self.snake1_length):
            if i == 0:
                # ============= 蛇头位置处理 =============
                # 蛇头始终位于轨迹的最前端(距离为0)
                # 这确保了蛇头完全跟随鼠标移动
                distance = 0
            else:
                # ============= 蛇身节段位置计算 =============
                distance = (i * 0.8 + 0.2) * SEGMENT_DISTANCE
            
            # ============= 获取节段在轨迹上的精确位置 =============
            # 调用轨迹定位算法，根据计算出的距离找到对应的坐标
            position = self.get_position_on_trail(distance, self.snake1_trail)
            
            # ============= 将位置添加到蛇身列表 =============
            # 按顺序构建完整的蛇身：[蛇头, 节段1, 节段2, ...]
            self.snake1.append(position)
    
    def update_snake2_body(self):
        """
        根据轨迹和蛇长度更新玩家2蛇身各节段位置 - 实现蛇头增长机制
        """
        # ============= 重置蛇身位置列表 =============
        # 清空当前蛇身，准备重新计算所有节段位置
        self.snake2 = []
        
        # ============= 逐节段计算位置 =============
        # 遍历蛇的每个节段，从蛇头(索引0)到蛇尾
        for i in range(self.snake2_length):
            if i == 0:
                # ============= 蛇头位置处理 =============
                # 蛇头始终位于轨迹的最前端(距离为0)
                distance = 0
            else:
                # ============= 蛇身节段位置计算 =============
                distance = (i * 0.8 + 0.2) * SEGMENT_DISTANCE
            
            # ============= 获取节段在轨迹上的精确位置 =============
            # 调用轨迹定位算法，根据计算出的距离找到对应的坐标
            position = self.get_position_on_trail(distance, self.snake2_trail)
            
            # ============= 将位置添加到蛇身列表 =============
            # 按顺序构建完整的蛇身：[蛇头, 节段1, 节段2, ...]
            self.snake2.append(position)
                
    def handle_events(self):
        """
        处理所有用户输入事件和系统事件
        
        这个方法负责处理游戏中的所有用户交互，包括：
        - 窗口关闭事件
        - 键盘输入(重启游戏、退出游戏)
        - 鼠标移动(核心游戏控制)
        
        返回:
            bool: True表示游戏应该继续运行，False表示应该退出
        """
        # 遍历所有待处理的pygame事件队列
        for event in pygame.event.get():
            # ============= 处理窗口关闭事件 =============
            if event.type == pygame.QUIT:
                # 用户点击窗口关闭按钮，返回False终止游戏循环
                return False
                
            # ============= 处理键盘按键事件 =============
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # ESC键：无条件退出游戏
                    return False
                elif event.key in [pygame.K_r, pygame.K_SPACE] and self.game_over:
                    # R键或空格键：仅在游戏结束时可以重启游戏
                    # 这个条件检查防止了游戏进行中误触重启
                    self.reset_game()
                    
                # ============= AI控制玩家2 =============
                # 玩家2现在完全由AI控制，无需键盘输入
                # AI将在update()方法中做出决策
                    
            # ============= 处理鼠标移动事件 =============
            elif event.type == pygame.MOUSEMOTION:
                if not self.game_over:
                    # 只有在游戏未结束时才处理鼠标移动
                    # 避免游戏结束后鼠标移动影响游戏状态
                    
                    # 标记游戏开始：第一次鼠标移动时自动开始游戏
                    # 这提供了直观的"移动鼠标开始游戏"体验
                    self.game_started = True
                    
                    # 更新鼠标位置：获取当前鼠标在窗口中的像素坐标
                    # 这个位置将在update()方法中作为蛇头的目标位置
                    self.mouse_pos = pygame.mouse.get_pos()
                    
        # 返回True表示游戏应该继续运行
        return True
        
    def update(self):
        """
        双蛇对战游戏主逻辑更新方法
        
        处理两条蛇的所有游戏状态变化：
        1. 玩家1(鼠标控制)和玩家2(WASD控制)的位置更新
        2. 边界碰撞检测
        3. 食物碰撞检测(支持双蛇竞争)
        4. 蛇与蛇碰撞检测
        5. 自身碰撞检测
        6. 胜负判断
        """
        # ============= 游戏状态检查 =============
        if self.game_over or not self.game_started:
            return
            
        # ============= 玩家1蛇头位置更新(鼠标控制) =============
        new_head1 = self.mouse_pos
        
        # 检查玩家1是否移动
        player1_moved = True
        if len(self.snake1_trail) > 0 and new_head1 == self.snake1_trail[0]:
            player1_moved = False
        
        # ============= 玩家2蛇头位置更新(AI控制-智能决策) =============
        # 获取当前时间
        current_time = pygame.time.get_ticks()
        
        # 记录游戏开始时间
        if self.game_started and self.game_start_time == 0:
            self.game_start_time = current_time
        
        # 计算动态AI移动间隔（随时间递减，速度递增）
        if self.game_start_time > 0:
            elapsed_seconds = (current_time - self.game_start_time) / 1000.0
            # 每10秒减速2%（间隔减少2%）
            speed_multiplier = self.speed_increase_rate ** (elapsed_seconds / 10.0)
            self.current_ai_interval = max(
                self.min_ai_interval,
                int(SNAKE2_MOVE_INTERVAL * speed_multiplier)
            )
        
        # 检查是否到了玩家2的移动时间（使用动态间隔）
        if current_time - self.snake2_last_move_time >= self.current_ai_interval:
            # AI决策：获取最优移动方向
            ai_direction = self.ai_controller.get_next_direction(
                my_snake=self.snake2.copy(),  # AI控制的蛇（玩家2）
                opponent_snake=self.snake1.copy(),  # 对手（玩家1）
                foods=self.foods.copy(),  # 食物列表
                current_direction=self.snake2_direction  # 当前方向
            )
            
            # 更新AI决定的方向
            self.snake2_direction = ai_direction
            
            # 如果AI决定移动
            if self.snake2_direction != (0, 0):
                # 时间到了且有方向，更新玩家2位置
                new_head2 = (
                    self.snake2_pos[0] + self.snake2_direction[0] * SEGMENT_DISTANCE,
                    self.snake2_pos[1] + self.snake2_direction[1] * SEGMENT_DISTANCE
                )
                self.snake2_pos = new_head2
                self.snake2_last_move_time = current_time
                player2_moved = True
                
                # AI开始移动后，游戏自动开始
                self.game_started = True
            else:
                # AI决定不移动
                new_head2 = self.snake2_pos
                player2_moved = False
        else:
            # 还没到移动时间，玩家2保持原位
            new_head2 = self.snake2_pos
            player2_moved = False
        
        # ============= 边界碰撞检测 =============
        boundary_left = CELL_SIZE // 2
        boundary_right = WINDOW_SIZE - CELL_SIZE // 2
        boundary_top = CELL_SIZE // 2
        boundary_bottom = WINDOW_SIZE - CELL_SIZE // 2
        
        # 检查玩家1边界碰撞(仅在移动时)
        if player1_moved and (new_head1[0] < boundary_left or new_head1[0] >= boundary_right or 
            new_head1[1] < boundary_top or new_head1[1] >= boundary_bottom):
            self.game_over = True
            self.winner = 2  # 玩家2获胜
            return
            
        # 检查玩家2边界碰撞(仅在移动时)
        if player2_moved and (new_head2[0] < boundary_left or new_head2[0] >= boundary_right or 
            new_head2[1] < boundary_top or new_head2[1] >= boundary_bottom):
            self.game_over = True
            self.winner = 1  # 玩家1获胜
            return
        
        # ============= 更新轨迹系统 =============
        # 玩家1轨迹更新(仅在移动时)
        if player1_moved:
            self.snake1_trail.insert(0, new_head1)
            max_trail_length1 = self.snake1_length * SEGMENT_DISTANCE + 100
            if len(self.snake1_trail) > max_trail_length1:
                self.snake1_trail = self.snake1_trail[:max_trail_length1]
        
        # 玩家2轨迹更新(仅在移动时)
        if player2_moved:
            self.snake2_trail.insert(0, new_head2)
            max_trail_length2 = self.snake2_length * SEGMENT_DISTANCE + 100
            if len(self.snake2_trail) > max_trail_length2:
                self.snake2_trail = self.snake2_trail[:max_trail_length2]
        
        # ============= 食物碰撞检测(双蛇竞争) =============
        eaten_food_indices = []  # 被吃掉的食物索引列表
        
        for i, food in enumerate(self.foods):
            # 检查玩家1是否吃到食物(仅在移动时)
            if player1_moved:
                food_distance1 = math.sqrt((new_head1[0] - food[0])**2 + (new_head1[1] - food[1])**2)
                if food_distance1 < CELL_SIZE:
                    eaten_food_indices.append((i, 1))  # (食物索引, 吃到的玩家)
                    continue
            
            # 检查玩家2是否吃到食物(仅在移动时)
            if player2_moved:
                food_distance2 = math.sqrt((new_head2[0] - food[0])**2 + (new_head2[1] - food[1])**2)
                if food_distance2 < CELL_SIZE:
                    eaten_food_indices.append((i, 2))  # (食物索引, 吃到的玩家)
        
        # 处理食物消耗(按索引倒序删除避免索引错位)
        for food_index, player in sorted(eaten_food_indices, reverse=True):
            self.foods.pop(food_index)
            if player == 1:
                self.score1 += 1
                self.snake1_length += 2
            else:
                self.score2 += 1
                self.snake2_length += 2
        
        # 如果有食物被吃，触发食物管理
        if eaten_food_indices:
            self.manage_food_count()
        
        # ============= 更新蛇身位置 =============
        self.update_snake_body()      # 玩家1
        self.update_snake2_body()     # 玩家2
        
        # ============= 自身碰撞检测 =============
        # 检查玩家1自身碰撞(仅在移动时)
        if player1_moved and len(self.snake1) > 4:
            for i in range(4, len(self.snake1)):
                collision_distance = math.sqrt(
                    (new_head1[0] - self.snake1[i][0])**2 + 
                    (new_head1[1] - self.snake1[i][1])**2
                )
                if collision_distance < CELL_SIZE * 0.7:
                    self.game_over = True
                    self.winner = 2  # 玩家2获胜
                    return
        
        # 检查玩家2自身碰撞(仅在移动时)
        if player2_moved and len(self.snake2) > 4:
            for i in range(4, len(self.snake2)):
                collision_distance = math.sqrt(
                    (new_head2[0] - self.snake2[i][0])**2 + 
                    (new_head2[1] - self.snake2[i][1])**2
                )
                if collision_distance < CELL_SIZE * 0.7:
                    self.game_over = True
                    self.winner = 1  # 玩家1获胜
                    return
        
        # ============= 蛇与蛇碰撞检测 =============
        # 检查玩家1蛇头是否撞到玩家2蛇身(仅在玩家1移动时)
        if player1_moved:
            for segment in self.snake2:
                collision_distance = math.sqrt(
                    (new_head1[0] - segment[0])**2 + 
                    (new_head1[1] - segment[1])**2
                )
                if collision_distance < CELL_SIZE * 0.7:
                    self.game_over = True
                    self.winner = 2  # 玩家2获胜
                    return
        
        # 检查玩家2蛇头是否撞到玩家1蛇身(仅在玩家2移动时)
        if player2_moved:
            for segment in self.snake1:
                collision_distance = math.sqrt(
                    (new_head2[0] - segment[0])**2 + 
                    (new_head2[1] - segment[1])**2
                )
                if collision_distance < CELL_SIZE * 0.7:
                    self.game_over = True
                    self.winner = 1  # 玩家1获胜
                    return
        
        # 检查蛇头相撞(平局) - 只有双方都移动时才检测
        if player1_moved and player2_moved:
            head_collision_distance = math.sqrt(
                (new_head1[0] - new_head2[0])**2 + 
                (new_head1[1] - new_head2[1])**2
            )
            if head_collision_distance < CELL_SIZE * 0.7:
                self.game_over = True
                self.winner = None  # 平局
                return
            
    def draw(self):
        """
        双蛇对战游戏渲染方法 - 绘制所有可视化元素
        
        这个方法负责将游戏的当前状态绘制到屏幕上，包括：
        1. 清空屏幕背景
        2. 绘制两条不同颜色的蛇(绿色和蓝色)
        3. 绘制所有食物
        4. 显示双方分数和游戏信息
        5. 显示游戏状态提示(开始提示、游戏结束界面)
        
        渲染顺序很重要：后绘制的元素会覆盖先绘制的元素。
        """
        # ============= 清空屏幕背景 =============
        # 使用黑色填充整个游戏窗口，为新一帧的绘制做准备
        # 这相当于"擦黑板"，清除上一帧的所有内容
        self.screen.fill(BLACK)
        
        # ============= 绘制玩家1蛇身系统(绿色) =============
        # 按照从蛇头到蛇尾的顺序绘制每个身体节段
        for i, segment in enumerate(self.snake1):
            # 根据节段位置决定颜色：蛇头使用深绿色，蛇身使用亮绿色
            # 这种颜色区分帮助玩家快速识别蛇头位置
            if i == 0:
                color = DARK_GREEN  # 蛇头：深绿色，更显眼
            else:
                color = GREEN       # 蛇身：亮绿色，与蛇头区分
            
            # 绘制圆形节段：以节段中心为圆心，半个CELL_SIZE为半径
            # 使用-1的边框让圆形之间有细微间隙，增强视觉效果
            pygame.draw.circle(self.screen, color, 
                             (int(segment[0]), int(segment[1])),  # 圆心坐标(转为整数像素)
                             CELL_SIZE // 2 - 1)                 # 半径(留1像素间隙)
        
        # ============= 绘制玩家2蛇身系统(蓝色) =============
        # 按照从蛇头到蛇尾的顺序绘制每个身体节段
        for i, segment in enumerate(self.snake2):
            # 根据节段位置决定颜色：蛇头使用深蓝色，蛇身使用亮蓝色
            # 与玩家1形成鲜明对比，便于区分
            if i == 0:
                color = DARK_BLUE   # 蛇头：深蓝色，更显眼
            else:
                color = BLUE        # 蛇身：亮蓝色，与蛇头区分
            
            # 绘制圆形节段：与玩家1相同的绘制方式
            pygame.draw.circle(self.screen, color, 
                             (int(segment[0]), int(segment[1])),  # 圆心坐标(转为整数像素)
                             CELL_SIZE // 2 - 1)                 # 半径(留1像素间隙)
            
        # ============= 绘制食物系统 =============
        # 绘制屏幕上的所有食物
        for food in self.foods:
            # 食物使用红色圆形表示，与两条蛇形成鲜明对比
            # 大小与蛇身节段相同，方便玩家判断碰撞范围
            pygame.draw.circle(self.screen, RED, 
                             (int(food[0]), int(food[1])),    # 食物中心坐标
                             CELL_SIZE // 2 - 1)             # 与蛇身相同的半径
        
        # ============= 双方分数信息显示 =============
        # 显示玩家1分数(左上角)
        score1_text = self.font.render(f"Player1(Mouse): {self.score1}", True, GREEN)
        self.screen.blit(score1_text, (10, 10))  # 左上角位置显示分数
        
        # 显示AI玩家分数(右上角)
        score2_text = self.font.render(f"AI Player: {self.score2}", True, BLUE)
        score2_rect = score2_text.get_rect()
        self.screen.blit(score2_text, (WINDOW_SIZE - score2_rect.width - 10, 10))  # 右上角位置
        
        # 显示当前食物数量和AI速度信息（开发调试信息）
        food_count_text = self.font.render(f"Foods: {len(self.foods)}", True, WHITE)
        self.screen.blit(food_count_text, (10, 50))  # 分数下方显示食物数量
        
        # 显示AI当前移动间隔（速度信息）
        ai_speed_text = self.font.render(f"AI Speed: {self.current_ai_interval}ms", True, WHITE)
        self.screen.blit(ai_speed_text, (10, 85))  # 显示AI当前决策间隔
        
        # ============= 游戏状态提示界面 =============
        # 游戏开始前的提示界面
        if not self.game_started and not self.game_over:
            # 显示人机对战游戏开始提示
            start_texts = [
                "Human vs AI Snake Battle!",
                "Player: Move mouse to control green snake",
                "AI: Controls blue snake automatically",
                "Move mouse to start the battle!"
            ]
            
            # 垂直居中显示多行提示文字
            total_height = len(start_texts) * 40  # 每行40像素高度
            start_y = (WINDOW_SIZE - total_height) // 2
            
            for i, text in enumerate(start_texts):
                rendered_text = self.font.render(text, True, WHITE)
                text_rect = rendered_text.get_rect(center=(WINDOW_SIZE // 2, start_y + i * 40))
                self.screen.blit(rendered_text, text_rect)
        
        # 游戏结束界面
        if self.game_over:
            # ============= 游戏结束信息准备 =============
            # 根据胜负情况创建不同的结束界面
            if self.winner == 1:
                game_over_text = self.font.render("Human Player Wins!", True, GREEN)
            elif self.winner == 2:
                game_over_text = self.font.render("AI Player Wins!", True, BLUE)
            else:
                game_over_text = self.font.render("Draw! Both players crashed!", True, WHITE)
            
            # 显示最终分数
            final_score_text = self.font.render(f"Final Scores - Human: {self.score1}  AI: {self.score2}", True, WHITE)
            
            # 重启说明
            restart_text = self.font.render("Press R or SPACE to restart", True, WHITE)
            quit_text = self.font.render("Press ESC to quit", True, WHITE)
            
            # 将所有文本组织成列表，便于批量处理
            texts = [game_over_text, final_score_text, restart_text, quit_text]
            
            # ============= 垂直居中布局计算 =============
            # 计算所有文本的总高度，用于垂直居中对齐
            total_height = sum(text.get_height() for text in texts) + len(texts) * 10  # 包含行间距
            start_y = (WINDOW_SIZE - total_height) // 2  # 起始Y坐标，实现垂直居中
            
            # ============= 逐行绘制游戏结束界面 =============
            for i, text in enumerate(texts):
                # 计算每行文本的位置：水平居中，垂直按顺序排列
                text_rect = text.get_rect(
                    center=(WINDOW_SIZE // 2,                           # 水平居中
                            start_y + i * (text.get_height() + 10))    # 垂直排列，10像素行间距
                )
                # 绘制文本到屏幕
                self.screen.blit(text, text_rect)
        
        # ============= 屏幕更新 =============
        # 将所有绘制内容从缓冲区刷新到实际屏幕上
        # 这是Pygame双缓冲机制的关键步骤，确保画面流畅无闪烁
        pygame.display.flip()
        
    def run(self):
        """
        游戏主循环 - 启动和维持游戏运行
        
        这是游戏的心脏，负责：
        1. 维持游戏的主要运行循环
        2. 协调事件处理、逻辑更新和屏幕渲染
        3. 控制游戏帧率以确保稳定性能
        4. 处理游戏退出和资源清理
        
        游戏循环遵循经典的"处理输入-更新逻辑-渲染画面"模式。
        """
        # ============= 游戏循环控制变量 =============
        # 主循环标志：True表示游戏继续运行，False表示退出游戏
        running = True
        
        # ============= 主游戏循环 =============
        # 这个循环会一直执行直到玩家退出游戏
        while running:
            # ============= 第一阶段：事件处理 =============
            # 处理所有用户输入和系统事件（鼠标、键盘、窗口关闭等）
            # handle_events()返回False时表示用户要求退出游戏
            running = self.handle_events()
            
            # ============= 第二阶段：游戏逻辑更新 =============
            # 更新游戏状态：蛇移动、碰撞检测、分数计算等
            # 这里包含了所有游戏机制的实现
            self.update()
            
            # ============= 第三阶段：屏幕渲染 =============
            # 将当前游戏状态绘制到屏幕上
            # 包括蛇身、食物、UI文本等所有可视元素
            self.draw()
            
            # ============= 第四阶段：帧率控制 =============
            # 限制游戏运行在10FPS，确保：
            # - 游戏速度稳定，不受硬件性能影响
            # - 鼠标跟随足够平滑
            # - CPU占用率保持在合理水平
            # - 为食物分布系统提供稳定的时间基准
            self.clock.tick(10)
            
        # ============= 游戏退出和资源清理 =============
        # 当游戏循环结束时，执行清理工作
        pygame.quit()  # 关闭pygame模块，释放所有资源
        sys.exit()     # 完全退出Python程序


# ============= 程序入口点 =============
if __name__ == "__main__":
    """
    主程序入口
    
    这段代码只有在直接运行此文件时才会执行（而不是被import时）。
    包含了错误处理机制，确保友好的错误提示。
    """
    try:
        # ============= 创建和启动游戏 =============
        # 实例化SnakeGame类，这会初始化所有游戏组件
        game = SnakeGame()
        
        # 启动游戏主循环，开始游戏体验
        game.run()
        
    except ImportError:
        # ============= 依赖库缺失处理 =============
        # 如果pygame库未安装，提供友好的安装提示
        print("Please install pygame library:")
        print("pip install pygame")
        sys.exit(1)  # 以错误代码1退出，表示安装依赖失败