# -*- coding: utf-8 -*-
"""
贪吃蛇AI选择器和游戏启动器
==========================

提供多种AI选项的游戏启动器：
1. 原版A*算法AI
2. 强化学习训练的AI
3. 智能随机AI
4. 纯随机AI

作者: Claude Code Assistant
"""

import os
import sys
from pathlib import Path

# 处理Windows控制台编码问题
def safe_print(text):
    """安全打印中文字符"""
    try:
        print(text)
    except UnicodeEncodeError:
        # 如果出现编码错误，转换为ASCII可显示字符
        text_ascii = text.encode('ascii', 'replace').decode('ascii')
        print(text_ascii)

def safe_input(prompt):
    """安全输入中文字符"""
    try:
        return input(prompt)
    except UnicodeDecodeError:
        return input(prompt.encode('ascii', 'replace').decode('ascii'))

def find_available_models():
    """查找可用的强化学习模型"""
    model_paths = []
    
    # 检查不同的模型目录
    model_dirs = [
        "snake_rl_models",
        "snake_rl_models_quick", 
        "snake_rl_models_demo"
    ]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            # 查找.zip文件（Stable-Baselines3模型）
            for file in os.listdir(model_dir):
                if file.endswith('.zip'):
                    model_paths.append(os.path.join(model_dir, file))
    
    return model_paths

def create_game_with_ai(ai_choice: str, model_path: str = None):
    """根据选择创建带有指定AI的游戏"""
    
    # 导入必要的模块
    import pygame
    import sys
    
    # 修改游戏类以支持不同的AI
    class ConfigurableSnakeGame:
        def __init__(self, ai_type="a_star", model_path=None):
            # 初始化pygame和基础设置
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            pygame.display.set_caption(f"Snake Game - {ai_type.upper()} AI")
            pygame.mouse.set_visible(False)
            self.clock = pygame.time.Clock()
            
            # 根据AI类型选择不同的AI控制器
            if ai_type == "rl":
                from snake_rl_ai import RLSnakeAI
                print(f"Loading RL AI, Model path: {model_path}")
                self.ai_controller = RLSnakeAI(600, 20, 15, model_path)
                self.ai_type = "Reinforcement Learning"
                
            elif ai_type == "random":
                from snake_random_ai import RandomSnakeAI
                print("Using Smart Random AI")
                self.ai_controller = RandomSnakeAI(600, 20, 15)
                self.ai_type = "Smart Random"
                
            else:  # 默认使用A*
                from snake_ai import SnakeAI
                print("Using A* Algorithm AI")
                self.ai_controller = SnakeAI(600, 20, 15)
                self.ai_type = "A* Algorithm"
            
            # 初始化字体
            try:
                self.font = pygame.font.Font("C:/Windows/Fonts/simsun.ttc", 36)
            except:
                self.font = pygame.font.Font(None, 36)
                
            # 从原始游戏复制初始化逻辑
            self.reset_game()
            
        def reset_game(self):
            """重置游戏状态"""
            center_x = 600 // 2
            center_y = 600 // 2
            
            # 玩家1蛇身系统初始化（鼠标控制）
            self.snake1_trail = [(center_x - 100, center_y)]
            self.snake1_length = 1
            self.snake1 = [(center_x - 100, center_y)]
            
            # 玩家2蛇身系统初始化（AI控制）
            self.snake2_trail = [(center_x + 100, center_y)]
            self.snake2_length = 1
            self.snake2 = [(center_x + 100, center_y)]
            self.snake2_direction = (0, 0)
            self.snake2_pos = (center_x + 100, center_y)
            self.snake2_last_move_time = 0
            
            # 控制系统
            self.mouse_pos = (center_x - 100, center_y)
            
            # 食物系统
            self.foods = []
            self.generate_foods(5)
            
            # 游戏状态
            self.score1 = 0
            self.score2 = 0
            self.game_over = False
            self.game_started = False
            self.winner = None
            
            # AI动态速度系统
            self.game_start_time = 0
            self.current_ai_interval = 80
            self.speed_increase_rate = 0.98
            self.min_ai_interval = 30
            
        def generate_food(self):
            """生成单个食物"""
            import math, random
            max_attempts = 100
            
            for attempt in range(max_attempts):
                food_x = random.randint(20, 580)
                food_y = random.randint(20, 580)
                food = (food_x, food_y)
                
                # 检查与蛇身的距离冲突
                too_close_to_snake = False
                for segment in self.snake1:
                    distance = math.sqrt((food[0] - segment[0])**2 + (food[1] - segment[1])**2)
                    if distance < 30:
                        too_close_to_snake = True
                        break
                if not too_close_to_snake:
                    for segment in self.snake2:
                        distance = math.sqrt((food[0] - segment[0])**2 + (food[1] - segment[1])**2)
                        if distance < 30:
                            too_close_to_snake = True
                            break
                
                # 检查与现有食物的距离冲突
                too_close_to_food = False
                for existing_food in self.foods:
                    distance = math.sqrt((food[0] - existing_food[0])**2 + (food[1] - existing_food[1])**2)
                    if distance < 40:
                        too_close_to_food = True
                        break
                
                if not too_close_to_snake and not too_close_to_food:
                    return food
            
            return (random.randint(20, 580), random.randint(20, 580))
        
        def generate_foods(self, count):
            """批量生成食物"""
            for _ in range(count):
                if len(self.foods) >= 10:
                    break
                new_food = self.generate_food()
                self.foods.append(new_food)
        
        def get_position_on_trail(self, distance, snake_trail):
            """轨迹跟随算法"""
            import math
            if not snake_trail or distance <= 0:
                return snake_trail[0] if snake_trail else (300, 300)
            
            current_distance = 0
            for i in range(len(snake_trail) - 1):
                point1 = snake_trail[i]
                point2 = snake_trail[i + 1]
                segment_distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
                
                if current_distance + segment_distance >= distance:
                    remaining_distance = distance - current_distance
                    ratio = remaining_distance / segment_distance if segment_distance > 0 else 0
                    x = point1[0] + (point2[0] - point1[0]) * ratio
                    y = point1[1] + (point2[1] - point1[1]) * ratio
                    return (x, y)
                
                current_distance += segment_distance
            
            return snake_trail[-1]
        
        def update_snake_body(self):
            """更新蛇身位置"""
            self.snake1 = []
            for i in range(self.snake1_length):
                if i == 0:
                    distance = 0
                else:
                    distance = (i * 0.8 + 0.2) * 15
                position = self.get_position_on_trail(distance, self.snake1_trail)
                self.snake1.append(position)
            
            self.snake2 = []
            for i in range(self.snake2_length):
                if i == 0:
                    distance = 0
                else:
                    distance = (i * 0.8 + 0.2) * 15
                position = self.get_position_on_trail(distance, self.snake2_trail)
                self.snake2.append(position)
        
        def handle_events(self):
            """处理事件"""
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key in [pygame.K_r, pygame.K_SPACE] and self.game_over:
                        self.reset_game()
                elif event.type == pygame.MOUSEMOTION:
                    if not self.game_over:
                        self.game_started = True
                        self.mouse_pos = pygame.mouse.get_pos()
            return True
        
        def update(self):
            """游戏逻辑更新"""
            import math
            if self.game_over or not self.game_started:
                return
                
            # 玩家1更新
            new_head1 = self.mouse_pos
            player1_moved = True
            if len(self.snake1_trail) > 0 and new_head1 == self.snake1_trail[0]:
                player1_moved = False
            
            # AI控制的玩家2更新
            current_time = pygame.time.get_ticks()
            
            if self.game_started and self.game_start_time == 0:
                self.game_start_time = current_time
            
            # 动态速度调整
            if self.game_start_time > 0:
                elapsed_seconds = (current_time - self.game_start_time) / 1000.0
                speed_multiplier = self.speed_increase_rate ** (elapsed_seconds / 10.0)
                self.current_ai_interval = max(
                    self.min_ai_interval,
                    int(80 * speed_multiplier)
                )
            
            player2_moved = False
            if current_time - self.snake2_last_move_time >= self.current_ai_interval:
                ai_direction = self.ai_controller.get_next_direction(
                    my_snake=self.snake2.copy(),
                    opponent_snake=self.snake1.copy(),
                    foods=self.foods.copy(),
                    current_direction=self.snake2_direction
                )
                
                self.snake2_direction = ai_direction
                
                if self.snake2_direction != (0, 0):
                    new_head2 = (
                        self.snake2_pos[0] + self.snake2_direction[0] * 15,
                        self.snake2_pos[1] + self.snake2_direction[1] * 15
                    )
                    self.snake2_pos = new_head2
                    self.snake2_last_move_time = current_time
                    player2_moved = True
                    self.game_started = True
                else:
                    new_head2 = self.snake2_pos
            else:
                new_head2 = self.snake2_pos
            
            # 边界碰撞检测
            if player1_moved and (new_head1[0] < 10 or new_head1[0] >= 590 or 
                new_head1[1] < 10 or new_head1[1] >= 590):
                self.game_over = True
                self.winner = 2
                return
                
            if player2_moved and (new_head2[0] < 10 or new_head2[0] >= 590 or 
                new_head2[1] < 10 or new_head2[1] >= 590):
                self.game_over = True
                self.winner = 1
                return
            
            # 更新轨迹
            if player1_moved:
                self.snake1_trail.insert(0, new_head1)
                if len(self.snake1_trail) > self.snake1_length * 15 + 100:
                    self.snake1_trail = self.snake1_trail[:self.snake1_length * 15 + 100]
            
            if player2_moved:
                self.snake2_trail.insert(0, new_head2)
                if len(self.snake2_trail) > self.snake2_length * 15 + 100:
                    self.snake2_trail = self.snake2_trail[:self.snake2_length * 15 + 100]
            
            # 食物碰撞检测
            eaten_food_indices = []
            for i, food in enumerate(self.foods):
                if player1_moved:
                    food_distance1 = math.sqrt((new_head1[0] - food[0])**2 + (new_head1[1] - food[1])**2)
                    if food_distance1 < 20:
                        eaten_food_indices.append((i, 1))
                        continue
                
                if player2_moved:
                    food_distance2 = math.sqrt((new_head2[0] - food[0])**2 + (new_head2[1] - food[1])**2)
                    if food_distance2 < 20:
                        eaten_food_indices.append((i, 2))
            
            # 处理食物消耗
            for food_index, player in sorted(eaten_food_indices, reverse=True):
                self.foods.pop(food_index)
                if player == 1:
                    self.score1 += 1
                    self.snake1_length += 2
                else:
                    self.score2 += 1
                    self.snake2_length += 2
            
            if eaten_food_indices:
                self.generate_foods(len(eaten_food_indices))
            
            # 更新蛇身
            self.update_snake_body()
            
            # 碰撞检测（完整版）
            # ============= 自身碰撞检测 =============
            if player1_moved and len(self.snake1) > 4:
                for i in range(4, len(self.snake1)):
                    collision_distance = math.sqrt(
                        (new_head1[0] - self.snake1[i][0])**2 + 
                        (new_head1[1] - self.snake1[i][1])**2
                    )
                    if collision_distance < 20 * 0.7:  # 使用与主游戏相同的碰撞阈值
                        self.game_over = True
                        self.winner = 2
                        return
            
            if player2_moved and len(self.snake2) > 4:
                for i in range(4, len(self.snake2)):
                    collision_distance = math.sqrt(
                        (new_head2[0] - self.snake2[i][0])**2 + 
                        (new_head2[1] - self.snake2[i][1])**2
                    )
                    if collision_distance < 20 * 0.7:  # 使用与主游戏相同的碰撞阈值
                        self.game_over = True
                        self.winner = 1
                        return
            
            # ============= 蛇与蛇碰撞检测 =============
            # 检查玩家1蛇头是否撞到玩家2蛇身
            if player1_moved:
                for segment in self.snake2:
                    collision_distance = math.sqrt(
                        (new_head1[0] - segment[0])**2 + 
                        (new_head1[1] - segment[1])**2
                    )
                    if collision_distance < 20 * 0.7:  # 使用与主游戏相同的碰撞阈值
                        self.game_over = True
                        self.winner = 2  # 玩家2获胜（玩家1撞到玩家2）
                        return
            
            # 检查玩家2蛇头是否撞到玩家1蛇身
            if player2_moved:
                for segment in self.snake1:
                    collision_distance = math.sqrt(
                        (new_head2[0] - segment[0])**2 + 
                        (new_head2[1] - segment[1])**2
                    )
                    if collision_distance < 20 * 0.7:  # 使用与主游戏相同的碰撞阈值
                        self.game_over = True
                        self.winner = 1  # 玩家1获胜（玩家2撞到玩家1）
                        return
            
            # ============= 蛇头相撞检测（平局）=============
            # 检查两条蛇的蛇头是否直接相撞
            if player1_moved and player2_moved:
                head_collision_distance = math.sqrt(
                    (new_head1[0] - new_head2[0])**2 + 
                    (new_head1[1] - new_head2[1])**2
                )
                if head_collision_distance < 20 * 0.7:  # 使用与主游戏相同的碰撞阈值
                    self.game_over = True
                    self.winner = None  # 平局
                    return
        
        def draw(self):
            """绘制游戏"""
            # 清空屏幕
            self.screen.fill((0, 0, 0))
            
            # 绘制蛇1（绿色）
            for i, segment in enumerate(self.snake1):
                color = (0, 150, 0) if i == 0 else (0, 255, 0)
                pygame.draw.circle(self.screen, color, 
                                 (int(segment[0]), int(segment[1])), 9)
            
            # 绘制蛇2（蓝色）
            for i, segment in enumerate(self.snake2):
                color = (0, 50, 150) if i == 0 else (0, 100, 255)
                pygame.draw.circle(self.screen, color, 
                                 (int(segment[0]), int(segment[1])), 9)
            
            # 绘制食物
            for food in self.foods:
                pygame.draw.circle(self.screen, (255, 0, 0), 
                                 (int(food[0]), int(food[1])), 9)
            
            # 显示分数
            score1_text = self.font.render(f"Human Player: {self.score1}", True, (0, 255, 0))
            self.screen.blit(score1_text, (10, 10))
            
            score2_text = self.font.render(f"{self.ai_type} AI: {self.score2}", True, (0, 100, 255))
            score2_rect = score2_text.get_rect()
            self.screen.blit(score2_text, (600 - score2_rect.width - 10, 10))
            
            # 显示AI类型
            ai_type_text = self.font.render(f"AI Type: {self.ai_type}", True, (255, 255, 255))
            self.screen.blit(ai_type_text, (10, 560))
            
            # 游戏状态提示
            if not self.game_started and not self.game_over:
                start_texts = [
                    f"Human vs {self.ai_type} AI Battle",
                    "Move mouse to control green snake",
                    f"{self.ai_type} AI controls blue snake automatically", 
                    "Move mouse to start the game!"
                ]
                
                total_height = len(start_texts) * 40
                start_y = (600 - total_height) // 2
                
                for i, text in enumerate(start_texts):
                    rendered_text = self.font.render(text, True, (255, 255, 255))
                    text_rect = rendered_text.get_rect(center=(300, start_y + i * 40))
                    self.screen.blit(rendered_text, text_rect)
            
            if self.game_over:
                if self.winner == 1:
                    game_over_text = self.font.render("Human Player Wins!", True, (0, 255, 0))
                elif self.winner == 2:
                    game_over_text = self.font.render(f"{self.ai_type} AI Wins!", True, (0, 100, 255))
                else:
                    game_over_text = self.font.render("Draw!", True, (255, 255, 255))
                
                final_score_text = self.font.render(f"Final Scores - Human: {self.score1}  AI: {self.score2}", True, (255, 255, 255))
                restart_text = self.font.render("Press R or SPACE to restart", True, (255, 255, 255))
                
                texts = [game_over_text, final_score_text, restart_text]
                total_height = sum(text.get_height() for text in texts) + len(texts) * 10
                start_y = (600 - total_height) // 2
                
                for i, text in enumerate(texts):
                    text_rect = text.get_rect(center=(300, start_y + i * (text.get_height() + 10)))
                    self.screen.blit(text, text_rect)
            
            pygame.display.flip()
        
        def run(self):
            """游戏主循环"""
            running = True
            while running:
                running = self.handle_events()
                self.update()
                self.draw()
                self.clock.tick(10)
            
            pygame.quit()
            sys.exit()
    
    return ConfigurableSnakeGame(ai_choice, model_path)

def create_random_ai():
    """创建简单的随机AI作为后备选项"""
    random_ai_code = '''
"""
随机贪吃蛇AI控制器
"""
import random
import math
from typing import List, Tuple

class RandomSnakeAI:
    """
    随机行为的贪吃蛇AI，但会避免明显的危险
    """
    
    def __init__(self, window_size: int, cell_size: int, segment_distance: int):
        self.window_size = window_size
        self.cell_size = cell_size
        self.segment_distance = segment_distance
        
        # 8个方向
        self.directions = [
            (0, -1), (0, 1), (-1, 0), (1, 0),  # 上、下、左、右
            (-1, -1), (1, -1), (-1, 1), (1, 1)  # 左上、右上、左下、右下
        ]
        
    def get_next_direction(self, my_snake: List[Tuple[int, int]], 
                          opponent_snake: List[Tuple[int, int]],
                          foods: List[Tuple[int, int]], 
                          current_direction: Tuple[int, int]) -> Tuple[int, int]:
        """获取下一步移动方向"""
        if not my_snake:
            return current_direction
            
        my_head = my_snake[0]
        
        # 获取安全的方向
        safe_directions = []
        for direction in self.directions:
            next_pos = (
                my_head[0] + direction[0] * self.segment_distance,
                my_head[1] + direction[1] * self.segment_distance
            )
            
            if self._is_position_safe(next_pos, my_snake, opponent_snake):
                safe_directions.append(direction)
        
        # 如果有安全方向，随机选择一个
        if safe_directions:
            return random.choice(safe_directions)
        
        # 否则返回当前方向或随机方向
        return current_direction if current_direction != (0, 0) else random.choice(self.directions)
    
    def _is_position_safe(self, pos: Tuple[int, int], 
                         my_snake: List[Tuple[int, int]],
                         opponent_snake: List[Tuple[int, int]]) -> bool:
        """检查位置是否安全"""
        # 检查边界
        boundary_margin = self.cell_size // 2
        if (pos[0] < boundary_margin or 
            pos[0] >= self.window_size - boundary_margin or
            pos[1] < boundary_margin or 
            pos[1] >= self.window_size - boundary_margin):
            return False
            
        # 检查与自己身体的碰撞
        for segment in my_snake[1:]:
            distance = math.sqrt((pos[0] - segment[0])**2 + (pos[1] - segment[1])**2)
            if distance < self.cell_size * 0.8:
                return False
                
        # 检查与对手的碰撞
        for segment in opponent_snake:
            distance = math.sqrt((pos[0] - segment[0])**2 + (pos[1] - segment[1])**2)
            if distance < self.cell_size * 0.8:
                return False
                
        return True
    
    def get_status(self) -> str:
        return "RandomSnakeAI: Using random but safe movements"
'''
    
    # 保存随机AI文件
    with open('snake_random_ai.py', 'w') as f:
        f.write(random_ai_code)
    
    print("Random AI created: snake_random_ai.py")

def main():
    """主启动函数"""
    print("=" * 50)
    print("Snake AI Battle Game Launcher")
    print("=" * 50)
    print()
    
    print("Available AI Options:")
    print("1. A* Algorithm AI (Original pathfinding-based AI)")
    print("2. Reinforcement Learning AI (Requires pre-trained model)")
    print("3. Smart Random AI (Safe but random movements)")
    print("4. Train new Reinforcement Learning model")
    print("5. Exit")
    print()
    
    try:
        choice = input("Please choose AI type (1-5): ").strip()
        
        if choice == "1":
            print("\nLaunching A* Algorithm AI Battle...")
            game = create_game_with_ai("a_star")
            game.run()
            
        elif choice == "2":
            print("\nSearching for available RL models...")
            available_models = find_available_models()
            
            if not available_models:
                print("No trained RL models found")
                print("\nSuggestions:")
                print("1. Run: python quick_train.py to train new model")
                print("2. Or choose other AI types")
                return
            
            print(f"\nFound {len(available_models)} models:")
            for i, model_path in enumerate(available_models):
                print(f"{i+1}. {model_path}")
            
            try:
                model_choice = int(input(f"\nSelect model (1-{len(available_models)}): ")) - 1
                if 0 <= model_choice < len(available_models):
                    selected_model = available_models[model_choice]
                    print(f"\nLaunching RL AI Battle (Model: {selected_model})...")
                    game = create_game_with_ai("rl", selected_model)
                    game.run()
                else:
                    print("Invalid selection")
            except ValueError:
                print("Please enter a valid number")
                
        elif choice == "3":
            print("\nCreating Smart Random AI...")
            create_random_ai()
            print("\nLaunching Smart Random AI Battle...")
            game = create_game_with_ai("random")
            game.run()
            
        elif choice == "4":
            print("\nLaunching Reinforcement Learning Training...")
            import subprocess
            subprocess.run([sys.executable, "quick_train.py"])
            
        elif choice == "5":
            print("\nGoodbye!")
            
        else:
            print("Invalid choice, please enter 1-5")
            
    except KeyboardInterrupt:
        print("\n\nUser cancelled, goodbye!")
    except Exception as e:
        print(f"\nLaunch failed: {e}")
        print("\nPlease ensure all required files exist:")
        print("- snake_game.py")
        print("- snake_ai.py (A* algorithm)")
        print("- snake_rl_ai.py (Reinforcement Learning AI)")

if __name__ == "__main__":
    main()