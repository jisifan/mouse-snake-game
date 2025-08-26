# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Game

To run the multiplayer snake battle game:
```bash
python snake_game.py
```

**Dependencies**: Requires pygame library. If not installed, run `pip install pygame`

## Game Architecture

This is a **multiplayer snake battle game** built with pygame featuring two different control schemes and competitive gameplay.

### Core Architecture

**Single-file Structure**: The entire game is contained in `snake_game.py` with the `SnakeGame` class handling all game logic.

**Key Components**:
- **Player 1 (Green Snake)**: Mouse-controlled with trail-following movement system
- **Player 2 (Blue Snake)**: WASD keyboard-controlled with traditional directional movement  
- **Dual Trail Systems**: Both snakes use trail-following for smooth body movement
- **Competitive Food System**: Both players compete for the same food resources
- **Advanced Collision Detection**: Snake vs snake, self-collision, and boundary collision

### Critical Implementation Details

**Dual Control Systems**:
- **Player 1**: Snake head follows mouse exactly; body segments follow the head's historical trail
- **Player 2**: WASD keys control direction; snake moves at 5 pixels per frame with trail recording

**Trail-Following Logic**: 
- `snake1_trail` and `snake2_trail` lists record complete movement history of each snake head
- Each body segment positioned at fixed distance intervals along respective trails
- `get_position_on_trail(distance, snake_trail)` performs linear interpolation for precise positioning
- `SEGMENT_DISTANCE = 15` pixels ensures consistent body segment spacing for both snakes

**Competitive Food System**:
- Same weighted probability system maintains 3-10 foods simultaneously
- Both snakes can compete for the same food items
- Food collision detection runs for both players each frame
- Score and growth are tracked separately: `score1`/`score2`, `snake1_length`/`snake2_length`

**Growth Mechanism**: 
- Both snakes grow from head (not tail) by increasing respective snake lengths
- Each food adds 2 segments and 1 point to the consuming player
- Growth uses compressed distance calculation: `(i * 0.8 + 0.2) * SEGMENT_DISTANCE`

**Advanced Collision Detection**:
- **Self-Collision**: Each snake can crash into its own body (skips first 4 segments)
- **Snake vs Snake**: Either snake head can crash into the other snake's body
- **Head-to-Head**: Direct head collision results in a draw
- **Boundary Collision**: Either snake hitting walls ends the game

**Game End Conditions**:
- Player crashes into boundary → Other player wins
- Player crashes into own body → Other player wins  
- Player crashes into opponent's body → Opponent wins
- Both snake heads collide → Draw/Tie game

### Rendering System

**Visual Differentiation**:
- **Player 1**: Green snake (DARK_GREEN head, GREEN body)
- **Player 2**: Blue snake (DARK_BLUE head, BLUE body)
- **Food**: Red circles (same as original)

**UI Display**:
- Player 1 score shown in left corner (green text)
- Player 2 score shown in right corner (blue text)  
- Food count displayed for debugging
- Winner announcement with appropriate colors

## Game Controls

### Player 1 (Green Snake)
- **Mouse Movement**: Snake head follows mouse cursor exactly with no speed limits
- **Static Mouse**: Snake remains stationary

### Player 2 (Blue Snake)  
- **W Key**: Move up
- **S Key**: Move down
- **A Key**: Move left
- **D Key**: Move right

### Global Controls
- **R or SPACE**: Restart game (when game over)
- **ESC**: Quit game

## Key Constants

- `WINDOW_SIZE = 600`: Game window dimensions
- `CELL_SIZE = 20`: Base size for snake segments and food circles
- `SEGMENT_DISTANCE = 15`: Fixed distance between snake body segments
- Game runs at 10 FPS via `clock.tick(10)`
- Food count range: 3-10 simultaneous foods
- Score: 1 point per food consumed per player
- Movement: Player 2 moves 5 pixels per frame

## Development Notes

When modifying the dual-snake system, be aware that:
1. Both trail systems must be maintained independently with separate memory management
2. Collision detection now involves 6 different collision types (self×2, snake-vs-snake×2, boundary×2)
3. Food generation must account for both snake positions to avoid conflicts
4. Game state management includes winner tracking (1, 2, or None for draw)
5. Rendering system draws both snakes with distinct colors and shows dual scores
6. Event handling processes both mouse (Player 1) and keyboard (Player 2) input

## Competitive Balance

The game is designed with different control schemes to create interesting gameplay dynamics:
- **Mouse Control (P1)**: Offers precise control and quick reactions but requires constant hand movement
- **Keyboard Control (P2)**: Provides consistent movement speed and rhythm but less precision
- Both control methods have strategic advantages depending on the situation
- ultrathink 将玩家2的移动变成ai控制，要求固定速度，自动吃苹果并避免撞到自己、撞到玩家、撞到墙，用一个新的python文件来写这个AI，并训练他，然后在snake_game.py中调用这个ai来控制玩家2移动。
- 你的AI现在只左右动，我需要AI具备8个方向的移动能力。提高AI决策频率，AI要去抢果子吃，并且移动速度随着时间慢慢提升，争取单位时间内吃更多果子
- AI在一个角落里来回走，请用合适的强化学习训练框架，先训练两条AI蛇互相对抗，然后再把训练完成的AI拿来和玩家对抗。现在先把重点放在对抗强化学习AI的实现上。