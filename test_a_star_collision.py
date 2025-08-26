# -*- coding: utf-8 -*-
"""
A*算法AI碰撞测试
"""

try:
    from game_launcher import create_game_with_ai
    print("=== A*算法AI碰撞测试 ===")
    print("测试A*算法AI的碰撞检测功能...")
    print("\n现在启动游戏，请测试：")
    print("1. 移动鼠标让绿色蛇接近蓝色AI蛇")
    print("2. 尝试让两条蛇发生碰撞")
    print("3. 验证碰撞后游戏是否正确结束")
    print("\n开始测试...")
    
    game = create_game_with_ai("a_star")
    game.run()
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()