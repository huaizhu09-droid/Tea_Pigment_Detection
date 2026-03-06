#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
茶色素智能检测系统 - 主程序入口
Tea Pigment Intelligent Detection System
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui.main_window import MainWindow


def main():
    # 启用高DPI缩放，适配高分屏
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("茶色素智能检测系统")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("TeaTech")
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
