# -*- coding: utf-8 -*-
"""
直接测试A*算法AI的脚本
"""

try:
    from game_launcher import create_game_with_ai
    print("Testing A* Algorithm AI...")
    game = create_game_with_ai("a_star")
    game.run()
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()