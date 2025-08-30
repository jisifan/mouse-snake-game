# -*- coding: utf-8 -*-
"""
强化学习贪吃蛇游戏启动器
======================

专用的强化学习AI对战游戏系统，使用训练好的PPO模型。
"""

import pygame
import sys
import math
import random
import os
from typing import List, Tuple

class RLSnakeGame:
    """强化学习AI贪吃蛇游戏"""
    
    def __init__(self, model_path: str = "snake_rl_models_efficient/final_agent2.zip"):
        """
        初始化游戏
        
        参数:
            model_path: 强化学习模型路径
        """
        # 完整初始化pygame
        pygame.init()
        pygame.key.set_repeat()  # 禁用按键重复
        
        # 创建游戏窗口并确保焦点
        self.screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption("Snake RL Battle - 强化学习AI对战")
        
        # 确保窗口获得焦点并可以接收键盘输入
        pygame.mouse.set_visible(False)
        
        # 强制窗口获得焦点（Windows系统）
        import os
        if os.name == 'nt':  # Windows
            import ctypes
            from ctypes import wintypes
            hwnd = pygame.display.get_wm_info()["window"]
            # 使用多种方法确保窗口获得焦点
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            ctypes.windll.user32.SetActiveWindow(hwnd)
            ctypes.windll.user32.SetFocus(hwnd)
            ctypes.windll.user32.BringWindowToTop(hwnd)
            
        # 刷新事件队列
        pygame.event.clear()
        pygame.event.pump()
        
        self.clock = pygame.time.Clock()
        
        # 加载强化学习AI
        from snake_rl_ai import RLSnakeAI
        if not os.path.exists(model_path):
            model_path = "snake_rl_models_efficient/agent2/model_100000.zip"  # 高效AI备用模型
        
        print(f"Loading RL AI model: {model_path}")
        try:
            self.ai_controller = RLSnakeAI(600, 20, 15, model_path)
            print("AI controller successfully created!")
        except Exception as e:
            print(f"Failed to create AI controller: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 游戏状态
        print("Initializing game state...")
        try:
            self.init_game_state()
            print("Game state initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize game state: {e}")
            import traceback
            traceback.print_exc()
            return
        
    def init_game_state(self):
        """初始化游戏状态"""
        print("    -> Resetting snake positions...")
        # 玩家蛇（绿色）- 鼠标控制
        self.snake1_trail = [(250, 300)]
        self.snake1_length = 1
        self.snake1 = [(250, 300)]
        self.last_mouse_pos = (250, 300)
        
        # AI蛇（蓝色）- 强化学习控制  
        self.snake2_trail = [(350, 300)]
        self.snake2_length = 1
        self.snake2 = [(350, 300)]
        self.ai_direction = (0, 0)
        
        print("    -> Resetting game state...")
        # 游戏设置
        self.foods = []
        self.generate_foods(5)
        self.game_over = False
        self.winner = None
        self.score1 = 0
        self.score2 = 0
        
        print("    -> Resetting AI timer...")
        # AI控制定时器
        self.last_ai_move = pygame.time.get_ticks()
        self.ai_interval = 80  # AI移动间隔（毫秒）
        
        print("    -> Resetting AI internal state...")
        # 重置AI内部状态
        if hasattr(self, 'ai_controller'):
            self.ai_controller.reset()
        
        print("    -> init_game_state() finished")
    
    def restart_game(self):
        """完全重启游戏"""
        print(">>> ENTERING restart_game() function")
        
        # 重置所有游戏状态
        print(">>> Calling init_game_state()...")
        self.init_game_state()
        print(">>> init_game_state() completed")
        
        # 强制重置鼠标
        print(">>> Resetting mouse position...")
        pygame.mouse.set_pos((250, 300))
        
        # 清除所有pygame事件
        print(">>> Clearing pygame events...")
        pygame.event.clear()
        
        print(">>> RESTART_GAME() COMPLETED SUCCESSFULLY!")
        
    def generate_foods(self, count):
        """生成指定数量的食物"""
        for _ in range(count):
            if len(self.foods) >= 10:
                break
            food = self.generate_single_food()
            if food:
                self.foods.append(food)
    
    def generate_single_food(self):
        """生成单个食物"""
        max_attempts = 100
        for attempt in range(max_attempts):
            food_x = random.randint(20, 580)
            food_y = random.randint(20, 580)
            food = (food_x, food_y)
            
            # 检查与蛇身的距离冲突
            too_close = False
            for segment in self.snake1 + self.snake2:
                distance = math.sqrt((food[0] - segment[0])**2 + (food[1] - segment[1])**2)
                if distance < 30:
                    too_close = True
                    break
            
            if not too_close:
                # 检查与已有食物的距离
                for existing_food in self.foods:
                    distance = math.sqrt((food[0] - existing_food[0])**2 + (food[1] - existing_food[1])**2)
                    if distance < 40:
                        too_close = True
                        break
            
            if not too_close:
                return food
        return None
    
    def update_snake_positions(self):
        """更新蛇的位置"""
        current_time = pygame.time.get_ticks()
        
        # 更新玩家蛇（鼠标控制）
        mouse_pos = pygame.mouse.get_pos()
        if mouse_pos != self.last_mouse_pos:
            self.snake1_trail.insert(0, mouse_pos)
            self.last_mouse_pos = mouse_pos
            
        # 限制轨迹长度
        max_trail_length = max(200, self.snake1_length * 15 + 50)
        if len(self.snake1_trail) > max_trail_length:
            self.snake1_trail = self.snake1_trail[:max_trail_length]
            
        # 更新AI蛇
        if current_time - self.last_ai_move > self.ai_interval:
            # 获取AI决策
            self.ai_direction = self.ai_controller.get_next_direction(
                self.snake2, self.snake1, self.foods, self.ai_direction
            )
            
            # 移动AI蛇头
            if self.ai_direction != (0, 0):
                new_head = (
                    self.snake2_trail[0][0] + self.ai_direction[0] * 15,
                    self.snake2_trail[0][1] + self.ai_direction[1] * 15
                )
                self.snake2_trail.insert(0, new_head)
                
                # 限制AI轨迹长度
                max_ai_trail = max(200, self.snake2_length * 15 + 50)
                if len(self.snake2_trail) > max_ai_trail:
                    self.snake2_trail = self.snake2_trail[:max_ai_trail]
                    
            self.last_ai_move = current_time
        
        # 更新蛇身位置
        self.update_snake_body(1)
        self.update_snake_body(2)
        
    def update_snake_body(self, snake_id):
        """更新指定蛇的身体位置"""
        if snake_id == 1:
            trail, length = self.snake1_trail, self.snake1_length
            self.snake1 = []
            for i in range(length):
                distance = (i * 0.8 + 0.2) * 15 if i > 0 else 0
                position = self.get_position_on_trail(distance, trail)
                self.snake1.append(position)
        else:
            trail, length = self.snake2_trail, self.snake2_length
            self.snake2 = []
            for i in range(length):
                distance = (i * 0.8 + 0.2) * 15 if i > 0 else 0
                position = self.get_position_on_trail(distance, trail)
                self.snake2.append(position)
    
    def get_position_on_trail(self, distance, trail):
        """在轨迹上根据距离找到位置"""
        if not trail or distance <= 0:
            return trail[0] if trail else (300, 300)
        
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
    
    def check_collisions(self):
        """检查碰撞"""
        if not self.snake1 or not self.snake2:
            return
            
        head1, head2 = self.snake1[0], self.snake2[0]
        
        # 检查边界碰撞
        boundary_margin = 10
        if (head1[0] < boundary_margin or head1[0] >= 590 or 
            head1[1] < boundary_margin or head1[1] >= 590):
            self.game_over = True
            self.winner = 2
            return
            
        if (head2[0] < boundary_margin or head2[0] >= 590 or 
            head2[1] < boundary_margin or head2[1] >= 590):
            self.game_over = True  
            self.winner = 1
            return
        
        # 检查蛇与蛇碰撞
        for segment in self.snake2:
            distance = math.sqrt((head1[0] - segment[0])**2 + (head1[1] - segment[1])**2)
            if distance < 20 * 0.7:
                self.game_over = True
                self.winner = 2
                return
                
        for segment in self.snake1:
            distance = math.sqrt((head2[0] - segment[0])**2 + (head2[1] - segment[1])**2)
            if distance < 20 * 0.7:
                self.game_over = True
                self.winner = 1
                return
        
        # 检查头部相撞
        head_distance = math.sqrt((head1[0] - head2[0])**2 + (head1[1] - head2[1])**2)
        if head_distance < 20 * 0.7:
            self.game_over = True
            self.winner = 0  # 平局
    
    def check_food_collision(self):
        """检查食物碰撞"""
        eaten_indices = []
        
        for i, food in enumerate(self.foods):
            # 玩家蛇吃食物
            if self.snake1:
                head1 = self.snake1[0]
                if math.sqrt((head1[0] - food[0])**2 + (head1[1] - food[1])**2) < 20:
                    eaten_indices.append(i)
                    self.snake1_length += 2
                    self.score1 += 1
                    continue
            
            # AI蛇吃食物
            if self.snake2:
                head2 = self.snake2[0]
                if math.sqrt((head2[0] - food[0])**2 + (head2[1] - food[1])**2) < 20:
                    eaten_indices.append(i)
                    self.snake2_length += 2
                    self.score2 += 1
        
        # 删除被吃的食物并生成新食物
        for i in sorted(eaten_indices, reverse=True):
            self.foods.pop(i)
        
        if eaten_indices:
            self.generate_foods(len(eaten_indices))
    
    def draw(self):
        """绘制游戏"""
        self.screen.fill((0, 0, 0))
        
        # 绘制玩家蛇（绿色）
        for i, segment in enumerate(self.snake1):
            color = (0, 150, 0) if i == 0 else (0, 255, 0)
            pygame.draw.circle(self.screen, color, (int(segment[0]), int(segment[1])), 9)
        
        # 绘制AI蛇（蓝色）
        for i, segment in enumerate(self.snake2):
            color = (0, 50, 150) if i == 0 else (0, 100, 255)
            pygame.draw.circle(self.screen, color, (int(segment[0]), int(segment[1])), 9)
        
        # 绘制食物（红色）
        for food in self.foods:
            pygame.draw.circle(self.screen, (255, 0, 0), (int(food[0]), int(food[1])), 9)
        
        # 绘制分数和信息
        font = pygame.font.Font(None, 36)
        player_text = font.render(f"Player: {self.score1}", True, (0, 255, 0))
        ai_text = font.render(f"RL AI: {self.score2}", True, (0, 100, 255))
        self.screen.blit(player_text, (10, 10))
        self.screen.blit(ai_text, (10, 50))
        
        # 游戏结束画面
        if self.game_over:
            overlay = pygame.Surface((600, 600))
            overlay.set_alpha(128)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, 0))
            
            result_font = pygame.font.Font(None, 72)
            if self.winner == 1:
                text = result_font.render("Player Wins!", True, (0, 255, 0))
            elif self.winner == 2:
                text = result_font.render("RL AI Wins!", True, (0, 100, 255))
            else:
                text = result_font.render("Draw!", True, (255, 255, 255))
                
            text_rect = text.get_rect(center=(300, 250))
            self.screen.blit(text, text_rect)
            
            restart_text = font.render("Press SPACE to restart, ESC to quit", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(300, 350))
            self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def run(self):
        """运行游戏主循环"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        print("RESTARTING GAME...")
                        self.restart_game()
                        print("RESTART COMPLETE!")
            
            if not self.game_over:
                self.update_snake_positions()
                self.check_collisions()
                self.check_food_collision()
            
            self.draw()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()

def create_rl_game(model_path: str = None):
    """创建强化学习AI游戏"""
    if model_path is None:
        # 使用最新训练的高效AI - 具备递进式里程碑奖励和效率导向行为
        model_path = "snake_rl_models_efficient/final_agent2.zip"
    
    return RLSnakeGame(model_path)

if __name__ == "__main__":
    print("=== 强化学习AI贪吃蛇对战 ===")
    print("使用全新高效AI训练系统的竞争性PPO强化学习AI")
    print("AI特点：递进式里程碑奖励、高效移动、积极吃食物！")
    print("移动鼠标控制绿色蛇，与全新高效AI激烈对战！")
    print("按ESC退出，游戏结束后按空格键重新开始")
    
    try:
        game = create_rl_game()
        print("Game object created, starting main loop...")
        game.run()
    except Exception as e:
        print(f"Game failed to start: {e}")
        import traceback
        traceback.print_exc()