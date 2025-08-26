# -*- coding: utf-8 -*-
"""
碰撞检测测试脚本
===============

测试修复后的蛇与蛇碰撞检测功能
"""

try:
    from game_launcher import create_game_with_ai
    print("=== 碰撞检测修复测试 ===")
    print("启动智能随机AI测试碰撞检测...")
    print("\n游戏说明：")
    print("- 移动鼠标控制绿色蛇")
    print("- 蓝色蛇由AI自动控制")
    print("- 试着让两条蛇相撞，验证游戏是否正确结束")
    print("- 撞到对方身体：对方获胜")
    print("- 蛇头相撞：平局")
    print("- 撞墙或撞自己：对方获胜")
    print("\n开始测试...")
    
    game = create_game_with_ai("random")
    game.run()
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()