import pygame
import random
import sys
import math

pygame.init()

WINDOW_SIZE = 600
CELL_SIZE = 20
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
        self.snake = [(center_x, center_y)]
        self.direction = (0, 0)
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
        self.game_started = False
        self.speed = 9  # 增加3倍速度
        self.pending_growth = 0  # 待增长的身体节段数
        
    def generate_food(self):
        while True:
            food_x = random.randint(CELL_SIZE, WINDOW_SIZE - CELL_SIZE)
            food_y = random.randint(CELL_SIZE, WINDOW_SIZE - CELL_SIZE)
            food = (food_x, food_y)
            # 确保食物不在蛇身上
            too_close = False
            for segment in self.snake:
                if math.sqrt((food[0] - segment[0])**2 + (food[1] - segment[1])**2) < CELL_SIZE * 1.5:
                    too_close = True
                    break
            if not too_close:
                return food
                
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
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    head_x, head_y = self.snake[0]
                    
                    # 计算从蛇头到鼠标的方向向量
                    dx = mouse_x - head_x
                    dy = mouse_y - head_y
                    
                    # 计算距离
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # 如果鼠标距离蛇头足够远，才更新方向
                    if distance > 10:
                        # 归一化方向向量并设置速度
                        self.direction = (dx/distance * self.speed, dy/distance * self.speed)
        return True
        
    def update(self):
        if self.game_over or not self.game_started:
            return
            
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # 检查边界碰撞 - 考虑蛇的半径
        if (new_head[0] < CELL_SIZE//2 or new_head[0] >= WINDOW_SIZE - CELL_SIZE//2 or 
            new_head[1] < CELL_SIZE//2 or new_head[1] >= WINDOW_SIZE - CELL_SIZE//2):
            self.game_over = True
            return
            
        # 先检查是否会吃到食物
        food_distance = math.sqrt((new_head[0] - self.food[0])**2 + (new_head[1] - self.food[1])**2)
        will_eat_food = food_distance < CELL_SIZE
        
        # 检查自身碰撞 - 跳过前3个身体节段(太近了，不可能是真碰撞)
        start_check = min(4, len(self.snake))
        for i in range(start_check, len(self.snake)):
            if math.sqrt((new_head[0] - self.snake[i][0])**2 + (new_head[1] - self.snake[i][1])**2) < CELL_SIZE * 0.7:
                self.game_over = True
                return
        
        # 移动蛇
        self.snake.insert(0, new_head)
        
        # 处理食物逻辑
        if will_eat_food:
            self.score += 10
            self.food = self.generate_food()
            # 果子直径约18像素，移动步长9像素，所以增长2节
            fruit_diameter = (CELL_SIZE // 2 - 1) * 2  # 18像素
            growth_segments = max(1, fruit_diameter // self.speed)  # 至少增长1节
            self.pending_growth += growth_segments
        
        # 处理身体增长：如果有待增长节段，就不移除尾巴
        if self.pending_growth > 0:
            self.pending_growth -= 1
        else:
            self.snake.pop()
            
    def draw(self):
        self.screen.fill(BLACK)
        
        # 绘制蛇身
        for i, segment in enumerate(self.snake):
            color = DARK_GREEN if i == 0 else GREEN
            # 以segment为中心绘制圆形
            pygame.draw.circle(self.screen, color, 
                             (int(segment[0]), int(segment[1])), CELL_SIZE // 2 - 1)
            
        # 绘制食物
        pygame.draw.circle(self.screen, RED, 
                         (int(self.food[0]), int(self.food[1])), CELL_SIZE // 2 - 1)
        
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
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