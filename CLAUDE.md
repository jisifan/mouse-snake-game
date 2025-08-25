# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Game

To run the snake game:
```bash
python snake_game.py
```

**Dependencies**: Requires pygame library. If not installed, run `pip install pygame`

## Game Architecture

This is a mouse-controlled snake game built with pygame featuring trail-following movement and intelligent multi-food distribution.

### Core Architecture

**Single-file Structure**: The entire game is contained in `snake_game.py` with the `SnakeGame` class handling all game logic.

**Key Components**:
- **Trail-Following System**: Snake head follows mouse exactly; body segments follow the head's historical trail
- **Multi-Food Management**: 3-10 foods simultaneously with weighted probability distribution
- **Trajectory-Based Movement**: Snake body positioning based on recorded movement trail with fixed segment distances
- **Smart Food Generation**: Intelligent spawning system maintaining normal distribution around 5-7 foods

### Critical Implementation Details

**Trail-Following Logic**: 
- `snake_trail` list records complete movement history of snake head
- Each body segment positioned at fixed distance intervals along the trail
- `get_position_on_trail(distance)` performs linear interpolation for precise positioning
- `SEGMENT_DISTANCE = 15` pixels ensures consistent body segment spacing

**Multi-Food System**:
- Weighted probability system: fewer foods → higher spawn chance, more foods → lower spawn chance  
- Food count distribution targets normal curve centered on 5-7 foods
- Each eaten food triggers 0-3 new food generation based on current count
- Prevents food clustering with minimum distance enforcement

**Growth Mechanism**: 
- Snake grows from head (not tail) by increasing `snake_length`
- New segments appear closer to head using compressed distance calculation: `(i * 0.8 + 0.2) * SEGMENT_DISTANCE`
- Each food adds 2 segments and 1 point (reduced from 10 points for better progression)

**Self-Collision Avoidance**: Skips collision checking for first 4 body segments due to trail-following creating very close segments during rapid movement.

**Memory Management**: Trail length automatically limited to prevent excessive memory usage while maintaining smooth movement.

### Food Distribution Algorithm

The `manage_food_count()` function uses carefully tuned weights:
- **3-4 foods**: Heavy bias toward generating more (weights favor 2-3 new foods)
- **5-6 foods**: Balanced generation (equal probability 0-3 new foods)  
- **7-8 foods**: Strong bias toward generating fewer (weights favor 0-1 new foods)
- **9-10 foods**: Extreme bias toward generating none (80%+ chance of 0 new foods)

## Game Controls

- **Mouse Movement**: Snake head follows mouse cursor exactly with no speed limits
- **R or SPACE**: Restart game (when game over)
- **ESC**: Quit game
- **Static Mouse**: Snake remains stationary

## Key Constants

- `WINDOW_SIZE = 600`: Game window dimensions
- `CELL_SIZE = 20`: Base size for snake segments and food circles
- `SEGMENT_DISTANCE = 15`: Fixed distance between snake body segments
- Game runs at 10 FPS via `clock.tick(10)`
- Food count range: 3-10 simultaneous foods
- Score: 1 point per food consumed

## Development Notes

When modifying the trail-following system, be aware that:
1. Trail memory management is critical - too short trails cause body positioning errors
2. Food generation weights directly impact gameplay balance and must be tested extensively  
3. Self-collision detection threshold may need adjustment if segment distance changes
4. Mouse sensitivity is 1:1 - any mouse movement immediately updates snake head position