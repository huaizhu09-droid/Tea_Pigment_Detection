# -*- coding: utf-8 -*-
"""
修复 Matplotlib 中文字体显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def fix_chinese_font():
    """配置 Matplotlib 使用中文字体"""
    
    # 尝试常见的中文字体
    font_names = [
        'SimHei',           # 黑体
        'Microsoft YaHei', # 微软雅黑
        'SimSun',           # 宋体
        'KaiTi',            # 楷体
        'FangSong',         # 仿宋
        'STXihei',          # 华文细黑
        'STKaiti',          # 华文楷体
        'STSong',           # 华文宋体
        'STFangsong',       # 华文仿宋
    ]
    
    # 查找可用的中文字体
    available_fonts = []
    for font_name in font_names:
        try:
            # 检查字体是否可用
            if len(fm.findfont(font_name)) > 0 and 'generic' not in fm.findfont(font_name):
                available_fonts.append(font_name)
                print(f"✓ 找到可用字体: {font_name}")
        except:
            pass
    
    if available_fonts:
        # 使用第一个可用的中文字体
        plt.rcParams['font.sans-serif'] = [available_fonts[0]]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print(f"✓ 已设置中文字体: {available_fonts[0]}")
        return True
    else:
        print("✗ 未找到可用的中文字体，图表中文可能显示为方框")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("修复 Matplotlib 中文字体显示")
    print("=" * 60)
    
    fix_chinese_font()
    
    print("\n" + "=" * 60)
    print("字体配置完成")
    print("=" * 60)
