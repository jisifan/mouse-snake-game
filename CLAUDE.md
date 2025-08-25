# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Game

To run the snake game:
```bash
python snake_game.py
```

**Dependencies**: Requires pygame library. If not installed, run `pip install pygame`

## Game Architecture

This is a mouse-controlled snake game built with pygame featuring continuous movement in any direction (not grid-based).

### Core Architecture

**Single-file Structure**: The entire game is contained in `snake_game.py` with the `SnakeGame` class handling all game logic.

**Key Components**:
- **Coordinate System**: Uses pixel-based coordinates (not grid), with snake segments positioned anywhere in the 600x600 window
- **Movement**: Snake follows mouse cursor with continuous directional movement at 9 pixels per frame
- **Collision Detection**: Uses Euclidean distance calculations for all collision detection (food, boundaries, self-collision)
- **Growth Mechanism**: Uses `pending_growth` counter to handle multi-segment growth over multiple frames

### Critical Implementation Details

**Self-Collision Logic**: The game skips checking collision with the first 4 body segments to prevent false positives during continuous movement (since segments are very close together with 9-pixel movement steps).

**Food Growth**: Each food item grows the snake by `fruit_diameter // speed` segments (currently 2 segments per food), distributed over multiple game frames.

**Boundary Checking**: Accounts for snake radius (`CELL_SIZE//2`) when checking window boundaries.

**Font Handling**: Attempts to load Chinese font from Windows system fonts, falls back to default pygame font if unavailable.

## Game Controls

- **Mouse Movement**: Controls snake direction
- **R or SPACE**: Restart game (when game over)
- **ESC**: Quit game

## Key Constants

- `WINDOW_SIZE = 600`: Game window dimensions
- `CELL_SIZE = 20`: Base size for snake segments and food
- `speed = 9`: Movement speed (pixels per frame)
- Game runs at 10 FPS via `clock.tick(10)`