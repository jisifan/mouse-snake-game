# -*- coding: utf-8 -*-
"""
直接测试智能随机AI的脚本
"""

try:
    from game_launcher import create_game_with_ai
    print("Testing Smart Random AI...")
    game = create_game_with_ai("random")
    game.run()
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()