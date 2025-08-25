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

# 初始化Pygame
pygame.init()

# ============= 游戏常量定义 =============
WINDOW_SIZE = 600           # 游戏窗口大小（像素）
CELL_SIZE = 20             # 蛇身和食物的基础大小（像素）
SEGMENT_DISTANCE = 15      # 蛇身节段间的固定距离（像素）
GRID_WIDTH = WINDOW_SIZE // CELL_SIZE    # 网格宽度（用于边界计算）
GRID_HEIGHT = WINDOW_SIZE // CELL_SIZE   # 网格高度（用于边界计算）

# ============= 颜色常量定义 =============
BLACK = (0, 0, 0)          # 背景色：黑色
GREEN = (0, 255, 0)        # 蛇身色：亮绿色
RED = (255, 0, 0)          # 食物色：红色
WHITE = (255, 255, 255)    # 文字色：白色
DARK_GREEN = (0, 150, 0)   # 蛇头色：暗绿色

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
        - 重置蛇的位置和长度
        - 清空移动轨迹
        - 重新生成食物
        - 重置分数和游戏状态标志
        """
        # 计算屏幕中心位置作为蛇的初始位置
        center_x = WINDOW_SIZE // 2
        center_y = WINDOW_SIZE // 2
        
        # ============= 蛇身系统初始化 =============
        # 轨迹跟随系统：记录蛇头的完整移动历史
        self.snake_trail = [(center_x, center_y)]  # 蛇头移动轨迹列表
        self.snake_length = 1                      # 当前蛇的长度（节数）
        self.snake = [(center_x, center_y)]        # 当前蛇身各节段的位置
        
        # ============= 控制和输入系统 =============
        self.mouse_pos = (center_x, center_y)      # 当前鼠标位置
        
        # ============= 食物系统初始化 =============
        self.foods = []                            # 食物列表（支持多个食物同时存在）
        self.generate_foods(5)                     # 初始生成5个食物
        
        # ============= 游戏状态变量 =============
        self.score = 0                             # 当前分数
        self.game_over = False                     # 游戏结束标志
        self.game_started = False                  # 游戏开始标志
        self.pending_growth = 0                    # 待增长的身体节段数（暂未使用）
        
    def generate_food(self):
        """
        生成单个食物
        
        在游戏区域内随机生成一个食物，确保：
        - 不与蛇身重叠（距离至少1.5倍CELL_SIZE）
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
            
            # 检查与蛇身的距离冲突
            too_close_to_snake = False
            for segment in self.snake:
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
    
    def get_position_on_trail(self, distance):
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
            
        返回:
            tuple: 对应距离处的(x, y)坐标位置
            
        边界处理:
        - 距离为0或负数时返回蛇头位置
        - 距离超出轨迹总长度时返回轨迹末端位置
        """
        # ============= 边界条件检查 =============
        if not self.snake_trail or distance <= 0:
            # 轨迹为空或距离无效时，返回蛇头位置或默认中心位置
            return self.snake_trail[0] if self.snake_trail else (WINDOW_SIZE//2, WINDOW_SIZE//2)
        
        # ============= 轨迹遍历和距离累积 =============
        current_distance = 0  # 当前已累积的距离
        
        # 遍历轨迹中的每个线段(相邻两点之间的连线)
        for i in range(len(self.snake_trail) - 1):
            point1 = self.snake_trail[i]      # 线段起点(距离蛇头更近)
            point2 = self.snake_trail[i + 1]  # 线段终点(距离蛇头更远)
            
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
        return self.snake_trail[-1]
    
    def update_snake_body(self):
        """
        根据轨迹和蛇长度更新蛇身各节段位置 - 实现蛇头增长机制
        
        这个方法是轨迹跟随系统的执行部分，负责将蛇身各节段精确定位在轨迹上：
        
        蛇头增长机制核心特点：
        1. 蛇头始终位于轨迹的起始点(距离=0)
        2. 新增长的节段出现在蛇头附近，而不是蛇尾
        3. 每个节段按递增距离分布，形成连续的蛇身
        4. 使用压缩系数(0.8)让新节段更紧密地跟随蛇头
        
        算法步骤：
        1. 清空当前蛇身位置列表
        2. 为每个节段计算在轨迹上的目标距离
        3. 调用get_position_on_trail()获取精确位置坐标
        4. 重建完整的蛇身位置列表
        
        距离分布公式：
        - 蛇头: distance = 0 (始终在轨迹起点)
        - 其他节段: distance = (i * 0.8 + 0.2) * SEGMENT_DISTANCE
          其中i为节段索引，0.8为压缩系数，0.2为偏移量
        
        增长效果：
        - 当蛇长度增加时，新节段出现在靠近蛇头的位置
        - 形成从头部向尾部的平滑延伸效果
        - 保持节段间的相对距离稳定
        """
        # ============= 重置蛇身位置列表 =============
        # 清空当前蛇身，准备重新计算所有节段位置
        self.snake = []
        
        # ============= 逐节段计算位置 =============
        # 遍历蛇的每个节段，从蛇头(索引0)到蛇尾
        for i in range(self.snake_length):
            if i == 0:
                # ============= 蛇头位置处理 =============
                # 蛇头始终位于轨迹的最前端(距离为0)
                # 这确保了蛇头完全跟随鼠标移动
                distance = 0
            else:
                # ============= 蛇身节段位置计算 =============
                # 使用特殊的距离分布公式实现蛇头增长效果：
                # 
                # 公式解析：(i * 0.8 + 0.2) * SEGMENT_DISTANCE
                # - i: 节段索引(1, 2, 3, ...)
                # - 0.8: 压缩系数，让节段更紧密分布
                # - 0.2: 基础偏移，防止节段重叠
                # - SEGMENT_DISTANCE: 基础间距(15像素)
                #
                # 实际距离序列：
                # 节段1: (1*0.8+0.2)*15 = 15像素
                # 节段2: (2*0.8+0.2)*15 = 27像素  
                # 节段3: (3*0.8+0.2)*15 = 39像素
                # 节段4: (4*0.8+0.2)*15 = 51像素
                # ...
                #
                # 这种分布的优势：
                # 1. 新节段出现在靠近蛇头的位置
                # 2. 节段间距离逐渐增加，形成自然的延伸效果
                # 3. 避免了传统蛇尾延长的不自然感
                distance = (i * 0.8 + 0.2) * SEGMENT_DISTANCE
            
            # ============= 获取节段在轨迹上的精确位置 =============
            # 调用轨迹定位算法，根据计算出的距离找到对应的坐标
            position = self.get_position_on_trail(distance)
            
            # ============= 将位置添加到蛇身列表 =============
            # 按顺序构建完整的蛇身：[蛇头, 节段1, 节段2, ...]
            self.snake.append(position)
                
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
        游戏主逻辑更新方法 - 处理所有游戏状态变化
        
        这是游戏的核心方法，每帧都会被调用，负责：
        1. 蛇头位置更新和轨迹记录
        2. 边界碰撞检测
        3. 食物碰撞检测和消耗
        4. 蛇身长度增长和分数更新
        5. 自身碰撞检测
        6. 蛇身位置重新计算
        
        方法会在游戏结束或未开始时直接返回，避免不必要的计算。
        """
        # ============= 游戏状态检查 =============
        # 如果游戏已结束或尚未开始，跳过所有更新逻辑
        # 这是性能优化，避免在非游戏状态下进行复杂计算
        if self.game_over or not self.game_started:
            return
            
        # ============= 蛇头位置更新 =============
        # 将鼠标当前位置设置为蛇头的新目标位置
        # 这实现了完全的鼠标跟随控制机制
        new_head = self.mouse_pos
        
        # ============= 移动检测优化 =============
        # 检查蛇头是否实际发生了位置变化
        # 如果鼠标没有移动，蛇保持静止状态，节省计算资源
        if len(self.snake_trail) > 0 and new_head == self.snake_trail[0]:
            return  # 鼠标位置未变化，跳过本次更新
        
        # ============= 边界碰撞检测 =============
        # 检查蛇头是否触碰到游戏区域边界
        # 使用CELL_SIZE//2作为碰撞边界，给蛇头留出半个身体的缓冲空间
        boundary_left = CELL_SIZE // 2        # 左边界
        boundary_right = WINDOW_SIZE - CELL_SIZE // 2  # 右边界  
        boundary_top = CELL_SIZE // 2         # 上边界
        boundary_bottom = WINDOW_SIZE - CELL_SIZE // 2  # 下边界
        
        if (new_head[0] < boundary_left or new_head[0] >= boundary_right or 
            new_head[1] < boundary_top or new_head[1] >= boundary_bottom):
            # 蛇头超出边界，游戏结束
            self.game_over = True
            return
        
        # ============= 轨迹系统更新 =============
        # 将蛇头的新位置添加到移动轨迹的最前端
        # 轨迹记录了蛇头的完整移动历史，是蛇身跟随系统的基础
        self.snake_trail.insert(0, new_head)
        
        # ============= 轨迹内存管理 =============
        # 限制轨迹长度以防止内存无限增长
        # 轨迹长度基于蛇身长度动态计算，保留足够的历史记录
        max_trail_length = self.snake_length * SEGMENT_DISTANCE + 100
        if len(self.snake_trail) > max_trail_length:
            # 截断过长的轨迹，保留最近的移动历史
            self.snake_trail = self.snake_trail[:max_trail_length]
        
        # ============= 食物碰撞检测 =============
        # 遍历所有食物，检测是否被蛇头吃到
        eaten_food_index = -1  # 被吃掉的食物索引，-1表示没有食物被吃
        
        for i, food in enumerate(self.foods):
            # 计算蛇头与食物中心的欧几里得距离
            food_distance = math.sqrt((new_head[0] - food[0])**2 + (new_head[1] - food[1])**2)
            
            # 如果距离小于一个CELL_SIZE，认为食物被吃到
            # 使用CELL_SIZE作为碰撞半径，提供合理的游戏手感
            if food_distance < CELL_SIZE:
                eaten_food_index = i  # 记录被吃掉的食物索引
                break  # 找到第一个被吃的食物就退出循环
        
        # ============= 食物消耗和奖励处理 =============
        if eaten_food_index >= 0:
            # 确实有食物被吃掉，执行相关奖励逻辑
            
            # 从食物列表中移除被吃掉的食物
            self.foods.pop(eaten_food_index)
            
            # 增加游戏分数：每个食物价值1分
            self.score += 1  
            
            # 蛇身长度增长：每吃一个食物增长2节
            # 这个数值影响游戏难度和成就感的平衡
            self.snake_length += 2  
            
            # 触发智能食物管理系统：根据当前食物数量随机生成新食物
            # 这维持了游戏中食物数量的动态平衡
            self.manage_food_count()
        
        # ============= 蛇身位置重计算 =============
        # 根据最新的轨迹和蛇长度，重新计算所有蛇身节段的位置
        # 这是轨迹跟随系统的核心执行部分
        self.update_snake_body()
        
        # ============= 自身碰撞检测 =============
        # 检查蛇头是否与自己的身体发生碰撞
        # 跳过前4节身体以避免误检测（蛇头附近的节段距离很近）
        if len(self.snake) > 4:
            for i in range(4, len(self.snake)):
                # 计算蛇头与每个身体节段的距离
                collision_distance = math.sqrt(
                    (new_head[0] - self.snake[i][0])**2 + 
                    (new_head[1] - self.snake[i][1])**2
                )
                
                # 使用0.7倍CELL_SIZE作为碰撞判定距离
                # 这个系数提供了合适的碰撞容错，避免过于严格的判定
                if collision_distance < CELL_SIZE * 0.7:
                    self.game_over = True  # 检测到自身碰撞，游戏结束
                    return
            
    def draw(self):
        """
        游戏渲染方法 - 绘制所有可视化元素
        
        这个方法负责将游戏的当前状态绘制到屏幕上，包括：
        1. 清空屏幕背景
        2. 绘制蛇身和蛇头
        3. 绘制所有食物
        4. 显示游戏信息(分数、食物数量)
        5. 显示游戏状态提示(开始提示、游戏结束界面)
        
        渲染顺序很重要：后绘制的元素会覆盖先绘制的元素。
        """
        # ============= 清空屏幕背景 =============
        # 使用黑色填充整个游戏窗口，为新一帧的绘制做准备
        # 这相当于"擦黑板"，清除上一帧的所有内容
        self.screen.fill(BLACK)
        
        # ============= 绘制蛇身系统 =============
        # 按照从蛇头到蛇尾的顺序绘制每个身体节段
        for i, segment in enumerate(self.snake):
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
            
        # ============= 绘制食物系统 =============
        # 绘制屏幕上的所有食物
        for food in self.foods:
            # 食物使用红色圆形表示，与蛇身形成鲜明对比
            # 大小与蛇身节段相同，方便玩家判断碰撞范围
            pygame.draw.circle(self.screen, RED, 
                             (int(food[0]), int(food[1])),    # 食物中心坐标
                             CELL_SIZE // 2 - 1)             # 与蛇身相同的半径
        
        # ============= 游戏信息显示 =============
        # 显示当前游戏分数
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))  # 左上角位置显示分数
        
        # 显示当前食物数量（开发调试信息，帮助验证食物管理系统）
        food_count_text = self.font.render(f"Foods: {len(self.foods)}", True, WHITE)
        self.screen.blit(food_count_text, (10, 50))  # 分数下方显示食物数量
        
        # ============= 游戏状态提示界面 =============
        # 游戏开始前的提示界面
        if not self.game_started and not self.game_over:
            # 显示"移动鼠标开始游戏"提示
            start_text = self.font.render("Move mouse to start", True, WHITE)
            # 将文字居中显示在屏幕中央
            text_rect = start_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
            self.screen.blit(start_text, text_rect)
        
        # 游戏结束界面
        if self.game_over:
            # ============= 游戏结束信息准备 =============
            # 创建游戏结束界面的所有文本元素
            game_over_text = self.font.render("Game Over!", True, WHITE)          # 游戏结束标题
            score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)  # 最终分数
            restart_text = self.font.render("Press R or SPACE to restart", True, WHITE)  # 重启说明
            quit_text = self.font.render("Press ESC to quit", True, WHITE)              # 退出说明
            
            # 将所有文本组织成列表，便于批量处理
            texts = [game_over_text, score_text, restart_text, quit_text]
            
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