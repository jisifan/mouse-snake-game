import pygame
import random
import sys
import math

pygame.init()

WINDOW_SIZE = 600
CELL_SIZE = 20
SEGMENT_DISTANCE = 15  # 每节身体间的固定距离
GRID_WIDTH = WINDOW_SIZE // CELL_SIZE
GRID_HEIGHT = WINDOW_SIZE // CELL_SIZE

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
DARK_GREEN = (0, 150, 0)

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Snake Game")
        pygame.mouse.set_visible(False)  # 隐藏鼠标指针
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.Font("C:/Windows/Fonts/simsun.ttc", 36)
        except:
            self.font = pygame.font.Font(None, 36)
        self.reset_game()
        
    def reset_game(self):
        center_x = WINDOW_SIZE // 2
        center_y = WINDOW_SIZE // 2
        
        # 蛇身系统：改为基于轨迹跟随
        self.snake_trail = [(center_x, center_y)]  # 蛇头移动轨迹
        self.snake_length = 1  # 蛇的长度（节数）
        self.snake = [(center_x, center_y)]  # 当前蛇身位置
        
        self.mouse_pos = (center_x, center_y)  # 鼠标位置
        self.foods = []  # 食物列表
        self.generate_foods(5)  # 初始生成5个食物
        self.score = 0
        self.game_over = False
        self.game_started = False
        self.pending_growth = 0  # 待增长的身体节段数
        
    def generate_food(self):
        """生成单个食物，避免与蛇身和现有食物重叠"""
        max_attempts = 100  # 防止无限循环
        for attempt in range(max_attempts):
            food_x = random.randint(CELL_SIZE, WINDOW_SIZE - CELL_SIZE)
            food_y = random.randint(CELL_SIZE, WINDOW_SIZE - CELL_SIZE)
            food = (food_x, food_y)
            
            # 检查与蛇身的距离
            too_close_to_snake = False
            for segment in self.snake:
                if math.sqrt((food[0] - segment[0])**2 + (food[1] - segment[1])**2) < CELL_SIZE * 1.5:
                    too_close_to_snake = True
                    break
            
            # 检查与现有食物的距离
            too_close_to_food = False
            for existing_food in self.foods:
                if math.sqrt((food[0] - existing_food[0])**2 + (food[1] - existing_food[1])**2) < CELL_SIZE * 2:
                    too_close_to_food = True
                    break
            
            if not too_close_to_snake and not too_close_to_food:
                return food
        
        # 如果尝试100次都没找到合适位置，返回一个随机位置
        return (random.randint(CELL_SIZE, WINDOW_SIZE - CELL_SIZE), 
                random.randint(CELL_SIZE, WINDOW_SIZE - CELL_SIZE))
    
    def generate_foods(self, count):
        """生成指定数量的食物"""
        for _ in range(count):
            if len(self.foods) >= 10:  # 最多10个食物
                break
            new_food = self.generate_food()
            self.foods.append(new_food)
    
    def manage_food_count(self):
        """管理食物数量：智能生成以达到正态分布"""
        current_count = len(self.foods)
        
        # 重新调整权重分布，减少大于7个的情况，增加小于5个的情况
        # 目标：让分布更加集中在5-7个果实
        
        if current_count <= 2:
            # 数量过少，强烈倾向生成更多
            weights = [5, 15, 40, 40]  # 生成0,1,2,3个的权重
        elif current_count == 3:
            # 偏少，倾向生成更多
            weights = [10, 20, 35, 35]
        elif current_count == 4:
            # 稍少，但要让它有机会减少到3个
            weights = [25, 30, 30, 15]  # 更平衡的分布
        elif current_count == 5:
            # 接近理想，但稍微倾向于减少
            weights = [30, 35, 25, 10]  # 倾向于生成更少
        elif current_count == 6:
            # 理想数量，均匀分布
            weights = [25, 25, 25, 25]
        elif current_count == 7:
            # 理想上限，强烈倾向于不生成太多
            weights = [45, 35, 15, 5]   # 强烈倾向0-1个
        elif current_count == 8:
            # 偏多，非常强烈倾向于不生成
            weights = [60, 30, 8, 2]    # 非常强烈倾向0个
        elif current_count >= 9:
            # 数量过多，极其强烈倾向于不生成
            weights = [80, 18, 2, 0]    # 几乎只生成0个
        
        # 使用加权随机选择生成数量
        choices = [0, 1, 2, 3]
        new_foods = random.choices(choices, weights=weights)[0]
        
        # 生成新果实
        self.generate_foods(new_foods)
        
        # 确保数量在3-10范围内（安全检查）
        final_count = len(self.foods)
        if final_count < 3:
            # 补充到3个
            needed = 3 - final_count
            self.generate_foods(needed)
        elif final_count > 10:
            # 删除多余的果实
            excess = final_count - 10
            for _ in range(excess):
                if self.foods:  # 确保列表不为空
                    # 随机删除一个果实
                    random_index = random.randint(0, len(self.foods) - 1)
                    self.foods.pop(random_index)
    
    def get_position_on_trail(self, distance):
        """根据距离在轨迹上找到对应位置"""
        if not self.snake_trail or distance <= 0:
            return self.snake_trail[0] if self.snake_trail else (WINDOW_SIZE//2, WINDOW_SIZE//2)
        
        current_distance = 0
        
        for i in range(len(self.snake_trail) - 1):
            point1 = self.snake_trail[i]
            point2 = self.snake_trail[i + 1]
            
            # 计算这两点间的距离
            segment_distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
            
            if current_distance + segment_distance >= distance:
                # 目标位置在这个线段上
                remaining_distance = distance - current_distance
                ratio = remaining_distance / segment_distance if segment_distance > 0 else 0
                
                # 在线段上插值找到精确位置
                x = point1[0] + (point2[0] - point1[0]) * ratio
                y = point1[1] + (point2[1] - point1[1]) * ratio
                return (x, y)
            
            current_distance += segment_distance
        
        # 如果距离超出轨迹长度，返回轨迹末端
        return self.snake_trail[-1]
    
    def update_snake_body(self):
        """根据轨迹和长度更新蛇身位置 - 蛇头增长版本"""
        self.snake = []
        
        for i in range(self.snake_length):
            if i == 0:
                # 蛇头始终在轨迹的起始位置
                distance = 0
            else:
                # 其他节段按递减的间隔分布，让新节段出现在头部附近
                # 使用更紧密的间隔来实现头部增长的视觉效果
                distance = (i * 0.8 + 0.2) * SEGMENT_DISTANCE
            
            position = self.get_position_on_trail(distance)
            self.snake.append(position)
                
    def handle_events(self):
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
        if self.game_over or not self.game_started:
            return
            
        # 蛇头跟随鼠标位置
        new_head = self.mouse_pos
        
        # 检查是否有实际移动
        if len(self.snake_trail) > 0 and new_head == self.snake_trail[0]:
            return  # 鼠标没移动，蛇保持静止
        
        # 检查边界碰撞
        if (new_head[0] < CELL_SIZE//2 or new_head[0] >= WINDOW_SIZE - CELL_SIZE//2 or 
            new_head[1] < CELL_SIZE//2 or new_head[1] >= WINDOW_SIZE - CELL_SIZE//2):
            self.game_over = True
            return
        
        # 将新的蛇头位置添加到轨迹前端
        self.snake_trail.insert(0, new_head)
        
        # 限制轨迹长度，避免内存过多占用
        max_trail_length = self.snake_length * SEGMENT_DISTANCE + 100
        if len(self.snake_trail) > max_trail_length:
            self.snake_trail = self.snake_trail[:max_trail_length]
        
        # 检查是否吃到食物
        eaten_food_index = -1
        for i, food in enumerate(self.foods):
            food_distance = math.sqrt((new_head[0] - food[0])**2 + (new_head[1] - food[1])**2)
            if food_distance < CELL_SIZE:
                eaten_food_index = i
                break
        
        # 处理食物逻辑
        if eaten_food_index >= 0:
            # 移除被吃掉的食物
            self.foods.pop(eaten_food_index)
            self.score += 1  # 每个果实+1分
            # 蛇头增长：增加蛇长度，新节段会出现在头部附近
            self.snake_length += 2  # 每个食物增长2节
            # 管理食物数量：随机生成0-3个新食物
            self.manage_food_count()
        
        # 根据轨迹更新蛇身位置
        self.update_snake_body()
        
        # 检查自身碰撞（跳过蛇头附近的几节）
        if len(self.snake) > 4:
            for i in range(4, len(self.snake)):
                if math.sqrt((new_head[0] - self.snake[i][0])**2 + (new_head[1] - self.snake[i][1])**2) < CELL_SIZE * 0.7:
                    self.game_over = True
                    return
            
    def draw(self):
        self.screen.fill(BLACK)
        
        # 绘制蛇身
        for i, segment in enumerate(self.snake):
            color = DARK_GREEN if i == 0 else GREEN
            # 以segment为中心绘制圆形
            pygame.draw.circle(self.screen, color, 
                             (int(segment[0]), int(segment[1])), CELL_SIZE // 2 - 1)
            
        # 绘制所有食物
        for food in self.foods:
            pygame.draw.circle(self.screen, RED, 
                             (int(food[0]), int(food[1])), CELL_SIZE // 2 - 1)
        
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # 显示当前食物数量
        food_count_text = self.font.render(f"Foods: {len(self.foods)}", True, WHITE)
        self.screen.blit(food_count_text, (10, 50))
        
        if not self.game_started and not self.game_over:
            start_text = self.font.render("Move mouse to start", True, WHITE)
            text_rect = start_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
            self.screen.blit(start_text, text_rect)
        
        if self.game_over:
            game_over_text = self.font.render("Game Over!", True, WHITE)
            score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
            restart_text = self.font.render("Press R or SPACE to restart", True, WHITE)
            quit_text = self.font.render("Press ESC to quit", True, WHITE)
            
            texts = [game_over_text, score_text, restart_text, quit_text]
            total_height = sum(text.get_height() for text in texts) + len(texts) * 10
            start_y = (WINDOW_SIZE - total_height) // 2
            
            for i, text in enumerate(texts):
                text_rect = text.get_rect(center=(WINDOW_SIZE // 2, start_y + i * (text.get_height() + 10)))
                self.screen.blit(text, text_rect)
        
        pygame.display.flip()
        
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(10)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    try:
        game = SnakeGame()
        game.run()
    except ImportError:
        print("Please install pygame library:")
        print("pip install pygame")
        sys.exit(1)