# -*- coding: utf-8 -*-
"""
茶色素智能检测系统 - 主程序
Tea Pigment Intelligent Detection System - Main Application
"""

import sys
import os
import json
import csv
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                             QFileDialog, QSlider, QComboBox, QGroupBox, 
                             QTextEdit, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QMessageBox, QTabWidget, QCheckBox,
                             QSpinBox, QDoubleSpinBox, QProgressBar, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.font_manager as fm

# 配置 Matplotlib 中文字体
def setup_chinese_font():
    """配置 Matplotlib 使用中文字体"""
    try:
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
                font_path = fm.findfont(font_name)
                if font_path and 'generic' not in font_path.lower():
                    available_fonts.append(font_name)
            except:
                pass
        
        if available_fonts:
            # 使用第一个可用的中文字体
            plt.rcParams['font.sans-serif'] = [available_fonts[0]]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        else:
            # 如果没有找到中文字体，使用默认字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    except Exception as e:
        print(f"字体配置失败: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 在导入其他模块后立即配置字体
setup_chinese_font()

from core.image_processor import ImageProcessor
from core.pigment_predictor_fixed import PigmentPredictor


class DetectionThread(QThread):
    """
    检测工作线程 - 在后台执行检测任务，避免阻塞UI
    """
    
    progress_update = pyqtSignal(int, str)
    detection_complete = pyqtSignal(dict)
    detection_error = pyqtSignal(str)
    
    def __init__(self, processor, predictor, image_path, use_calibration):
        super().__init__()
        self.processor = processor
        self.predictor = predictor
        self.image_path = image_path
        self.use_calibration = use_calibration
    
    def run(self):
        """执行检测任务"""
        try:
            self.progress_update.emit(10, "正在加载图像...")
            
            # 1. 加载图像
            img = self.processor.load_image(self.image_path)
            if img is None:
                raise ValueError("图像加载失败")
            
            self.progress_update.emit(30, "正在提取图像特征...")
            
            # 2. 提取图像特征
            features = self.processor.extract_color_features(img)
            
            self.progress_update.emit(50, "正在进行预测计算...")
            
            # 3. 调用预测模型
            predictions = self.predictor.predict(features)
            
            self.progress_update.emit(80, "正在分析结果...")
            
            # 4. 结果分析
            self.progress_update.emit(100, "检测完成")
            
            self.detection_complete.emit(predictions)
            
        except Exception as e:
            self.detection_error.emit(str(e))


class ResultVisualizationWidget(QWidget):
    """
    结果可视化组件 - 使用matplotlib绘制检测结果的柱状图
    """
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 创建matplotlib图表
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(self.canvas)
        
    def update_results(self, predictions: dict):
        """
        更新结果显示
        
        Args:
            predictions: 预测结果
        """
        try:
            self.figure.clear()
            
            # 创建子图
            ax = self.figure.add_subplot(111)
            
            # 准备数据
            pigments = ['茶黄素(TF)', '茶红素(TR)', '茶褐素(TB)']
            values = [predictions.get('tf', (0, 0))[0], 
                     predictions.get('tr', (0, 0))[0], 
                     predictions.get('tb', (0, 0))[0]]
            confidences = [predictions.get('tf', (0, 0))[1], 
                          predictions.get('tr', (0, 0))[1], 
                          predictions.get('tb', (0, 0))[1]]
            
            # 颜色设置
            colors = ['#FFD700', '#FF6347', '#8B4513']
            
            # 绘制柱状图
            bars = ax.bar(pigments, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # 在柱状图上方添加数值标签
            for bar, value, confidence in zip(bars, values, confidences):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}%\n(置信度:{confidence:.2f})',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 设置标题和标签
            ax.set_title('茶色素含量检测结果', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('含量 (%)', fontsize=12, fontweight='bold')
            ax.set_xlabel('茶色素类型', fontsize=12, fontweight='bold')
            
            # 设置y轴范围，从0开始
            ax.set_ylim(0, max(max(values) * 1.2, 1))
            
            # 添加网格
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # 美化边框
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1)
            
            # 调整布局
            self.figure.tight_layout()
            
            # 重绘图表
            self.canvas.draw()
            
        except Exception as e:
            print(f"结果可视化失败: {e}")


class MainWindow(QMainWindow):
    """
    主窗口类 - 实现GUI界面
    """
    
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.current_image = None
        self.processed_image = None
        self.current_predictions = None
        self.processed_images_count = 0
        
        # 初始化核心组件
        self.processor = ImageProcessor()
        self.predictor = PigmentPredictor()
        
        # 尝试加载模型
        self.load_models()
        
        # 初始化UI
        self.init_ui()
        
    def load_models(self):
        """加载预测模型"""
        try:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            if self.predictor.load_models(model_dir):
                print("模型加载成功")
            else:
                print("模型加载失败，将使用默认参数模式")
        except Exception as e:
            print(f"模型加载异常: {e}")
    
    def init_ui(self):
        """初始化用户界面"""
        # 设置窗口基本属性
        self.setWindowTitle("茶色素智能检测系统 v1.0")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # 设置整体布局
        self.setup_main_layout()
        
        # 应用主题颜色
        self.apply_theme()
        
    def setup_main_layout(self):
        """设置主界面布局"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 左侧控制面板
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=2)
        
        # 中间显示区域
        center_panel = self.create_center_panel()
        main_layout.addWidget(center_panel, stretch=6)
        
        # 右侧结果面板
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=2)
    
    def create_left_panel(self) -> QGroupBox:
        """创建左侧控制面板"""
        panel = QGroupBox("控制面板")
        panel.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 20px;
                background-color: #f5f5f5;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 文件操作区域
        file_group = self.create_file_operation_group()
        layout.addWidget(file_group)
        
        # 预处理参数区域
        preprocess_group = self.create_preprocess_group()
        layout.addWidget(preprocess_group)
        
        # 检测参数区域
        detection_group = self.create_detection_params_group()
        layout.addWidget(detection_group)
        
        # 校准区域
        calibration_group = self.create_calibration_group()
        layout.addWidget(calibration_group)
        
        # 系统信息区域
        system_group = self.create_system_info_group()
        layout.addWidget(system_group)
        
        layout.addStretch()
        
        return panel
    
    def create_file_operation_group(self) -> QGroupBox:
        """创建文件操作区域"""
        group = QGroupBox("文件操作")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        # 打开图片按钮
        self.btn_open_image = QPushButton("打开图片")
        self.btn_open_image.setMinimumHeight(45)
        self.btn_open_image.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.btn_open_image.clicked.connect(self.open_image)
        layout.addWidget(self.btn_open_image)
        
        # 开始检测按钮
        self.btn_start_detection = QPushButton("开始检测")
        self.btn_start_detection.setMinimumHeight(45)
        self.btn_start_detection.setEnabled(False)
        self.btn_start_detection.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.btn_start_detection.clicked.connect(self.start_detection)
        layout.addWidget(self.btn_start_detection)
        
        # 保存结果按钮
        self.btn_save_result = QPushButton("保存结果")
        self.btn_save_result.setMinimumHeight(45)
        self.btn_save_result.setEnabled(False)
        self.btn_save_result.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #EF6C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.btn_save_result.clicked.connect(self.save_result)
        layout.addWidget(self.btn_save_result)
        
        # 导出批量结果按钮
        self.btn_export_results = QPushButton("导出批量结果")
        self.btn_export_results.setMinimumHeight(45)
        self.btn_export_results.setEnabled(False)
        self.btn_export_results.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:pressed {
                background-color: #6A1B9A;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.btn_export_results.clicked.connect(self.export_results)
        layout.addWidget(self.btn_export_results)
        
        return group
    
    def create_preprocess_group(self) -> QGroupBox:
        """创建预处理参数区域"""
        group = QGroupBox("预处理参数")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        # 图像增强选择
        layout.addWidget(QLabel("图像增强:"))
        self.combo_enhancement = QComboBox()
        self.combo_enhancement.addItems(["无", "直方图均衡化", "CLAHE", "自适应直方图"])
        layout.addWidget(self.combo_enhancement)
        
        # 滤波方法选择
        layout.addWidget(QLabel("滤波方法:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["无", "高斯滤波", "中值滤波", "双边滤波"])
        layout.addWidget(self.combo_filter)
        
        # 颜色空间选择
        layout.addWidget(QLabel("主要颜色空间:"))
        self.combo_color_space = QComboBox()
        self.combo_color_space.addItems(["HSV", "L*a*b*", "RGB"])
        layout.addWidget(self.combo_color_space)
        
        return group
    
    def create_detection_params_group(self) -> QGroupBox:
        """创建检测参数区域"""
        group = QGroupBox("检测参数")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        # 置信度阈值
        layout.addWidget(QLabel("置信度阈值:"))
        self.slider_confidence = QSlider(Qt.Horizontal)
        self.slider_confidence.setRange(0, 100)
        self.slider_confidence.setValue(50)
        self.slider_confidence.valueChanged.connect(self.on_confidence_changed)
        layout.addWidget(self.slider_confidence)
        
        self.label_confidence_value = QLabel("50%")
        self.label_confidence_value.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_confidence_value)
        
        # 批量检测数量设置
        layout.addWidget(QLabel("批量检测数量:"))
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 100)
        self.spin_batch_size.setValue(10)
        layout.addWidget(self.spin_batch_size)
        
        return group
    
    def create_calibration_group(self) -> QGroupBox:
        """创建校准区域"""
        group = QGroupBox("颜色校准")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        # 启用校准复选框
        self.check_enable_calibration = QCheckBox("启用颜色校准")
        self.check_enable_calibration.setChecked(False)
        layout.addWidget(self.check_enable_calibration)
        
        # 校准色卡选择按钮
        self.btn_select_color_card = QPushButton("选择标准色卡")
        self.btn_select_color_card.setEnabled(False)
        self.btn_select_color_card.clicked.connect(self.select_color_card)
        layout.addWidget(self.btn_select_color_card)
        
        # 执行校准按钮
        self.btn_perform_calibration = QPushButton("执行校准")
        self.btn_perform_calibration.setEnabled(False)
        self.btn_perform_calibration.clicked.connect(self.perform_calibration)
        layout.addWidget(self.btn_perform_calibration)
        
        # 校准状态标签
        self.label_calibration_status = QLabel("校准状态: 未校准")
        self.label_calibration_status.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.label_calibration_status)
        
        # 绑定复选框事件
        self.check_enable_calibration.stateChanged.connect(self.on_calibration_checkbox_changed)
        
        return group
    
    def create_system_info_group(self) -> QGroupBox:
        """创建系统信息区域"""
        group = QGroupBox("系统信息")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        # 模型状态
        self.label_model_status = QLabel("模型状态: 未加载")
        layout.addWidget(self.label_model_status)
        
        # 处理图片数量
        self.label_processed_count = QLabel(f"已处理图片: {self.processed_images_count}")
        layout.addWidget(self.label_processed_count)
        
        # 当前图片信息
        self.label_current_image_info = QLabel("当前图片: 无")
        layout.addWidget(self.label_current_image_info)
        
        return group
    
    def create_center_panel(self) -> QWidget:
        """创建中间显示区域"""
        panel = QGroupBox("图像显示区域")
        panel.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #2196F3;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 20px;
                background-color: #ffffff;
            }
        """)
        
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 原始图像标签页
        self.tab_original = QWidget()
        self.setup_original_tab()
        self.tab_widget.addTab(self.tab_original, "原始图像")
        
        # 处理后图像标签页
        self.tab_processed = QWidget()
        self.setup_processed_tab()
        self.tab_widget.addTab(self.tab_processed, "处理后图像")
        
        layout.addWidget(self.tab_widget)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 20px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # 状态栏
        self.status_label = QLabel("请打开图片开始检测...")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #555555;
                font-size: 12px;
                padding: 5px;
                border: 1px solid #e0e0e0;
                border-radius: 3px;
                background-color: #f9f9f9;
            }
        """)
        layout.addWidget(self.status_label)
        
        return panel
    
    def setup_original_tab(self):
        """设置原始图像标签页"""
        layout = QVBoxLayout()
        self.tab_original.setLayout(layout)
        
        # 图像显示标签
        self.label_original_image = QLabel("暂无图像")
        self.label_original_image.setAlignment(Qt.AlignCenter)
        self.label_original_image.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 8px;
                background-color: #fafafa;
                color: #999999;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.label_original_image)
        
    def setup_processed_tab(self):
        """设置处理后图像标签页"""
        layout = QVBoxLayout()
        self.tab_processed.setLayout(layout)
        
        # 图像显示标签
        self.label_processed_image = QLabel("暂无处理后图像")
        self.label_processed_image.setAlignment(Qt.AlignCenter)
        self.label_processed_image.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 8px;
                background-color: #fafafa;
                color: #999999;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.label_processed_image)
    
    def create_right_panel(self) -> QGroupBox:
        """创建右侧结果面板"""
        panel = QGroupBox("检测结果")
        panel.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #FF9800;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 20px;
                background-color: #f5f5f5;
            }
        """)
        
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 数值结果显示区域
        numerical_group = self.create_numerical_results_group()
        layout.addWidget(numerical_group)
        
        # 可视化图表区域
        self.visualization_widget = ResultVisualizationWidget()
        layout.addWidget(self.visualization_widget)
        
        # 详细信息表格
        self.table_detailed_info = QTableWidget()
        self.setup_detailed_info_table()
        layout.addWidget(self.table_detailed_info)
        
        return panel
    
    def create_numerical_results_group(self) -> QGroupBox:
        """创建数值结果区域"""
        group = QGroupBox("茶色素含量")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        # 创建结果标签
        grid = QGridLayout()
        
        # 茶黄素结果
        grid.addWidget(QLabel("茶黄素(TF):"), 0, 0)
        self.label_tf_result = QLabel("-- %")
        self.label_tf_result.setStyleSheet("""
            QLabel {
                color: #FFD700;
                font-size: 18px;
                font-weight: bold;
                background-color: #ffffff;
                padding: 8px;
                border: 2px solid #FFD700;
                border-radius: 5px;
            }
        """)
        self.label_tf_result.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.label_tf_result, 0, 1)
        
        # 茶红素结果
        grid.addWidget(QLabel("茶红素(TR):"), 1, 0)
        self.label_tr_result = QLabel("-- %")
        self.label_tr_result.setStyleSheet("""
            QLabel {
                color: #FF6347;
                font-size: 18px;
                font-weight: bold;
                background-color: #ffffff;
                padding: 8px;
                border: 2px solid #FF6347;
                border-radius: 5px;
            }
        """)
        self.label_tr_result.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.label_tr_result, 1, 1)
        
        # 茶褐素结果
        grid.addWidget(QLabel("茶褐素(TB):"), 2, 0)
        self.label_tb_result = QLabel("-- %")
        self.label_tb_result.setStyleSheet("""
            QLabel {
                color: #8B4513;
                font-size: 18px;
                font-weight: bold;
                background-color: #ffffff;
                padding: 8px;
                border: 2px solid #8B4513;
                border-radius: 5px;
            }
        """)
        self.label_tb_result.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.label_tb_result, 2, 1)
        
        layout.addLayout(grid)
        
        # 总色素含量
        self.label_total_pigments = QLabel("总色素含量: -- %")
        self.label_total_pigments.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                background-color: #ffffff;
            }
        """)
        self.label_total_pigments.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_total_pigments)
        
        return group
    
    def setup_detailed_info_table(self):
        """设置详细信息表格"""
        # 设置表格列数
        self.table_detailed_info.setColumnCount(2)
        self.table_detailed_info.setHorizontalHeaderLabels(["项目", "值"])
        
        # 设置表格样式
        self.table_detailed_info.setStyleSheet("""
            QTableWidget {
                gridline-color: #e0e0e0;
                background-color: white;
                alternate-background-color: #f5f5f5;
            }
            QTableWidget::item {
                padding: 5px;
                border: none;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #388E3C;
            }
        """)
        
        # 设置列宽
        header = self.table_detailed_info.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        
        # 设置行高
        self.table_detailed_info.verticalHeader().setDefaultSectionSize(25)
        
        # 初始化表格内容
        self.table_detailed_info.setRowCount(12)
        self.table_detailed_info.setItem(0, 0, QTableWidgetItem("检测时间"))
        self.table_detailed_info.setItem(1, 0, QTableWidgetItem("图像尺寸"))
        self.table_detailed_info.setItem(2, 0, QTableWidgetItem("文件大小"))
        self.table_detailed_info.setItem(3, 0, QTableWidgetItem("茶黄素置信度"))
        self.table_detailed_info.setItem(4, 0, QTableWidgetItem("茶红素置信度"))
        self.table_detailed_info.setItem(5, 0, QTableWidgetItem("茶褐素置信度"))
        self.table_detailed_info.setItem(6, 0, QTableWidgetItem("平均置信度"))
        self.table_detailed_info.setItem(7, 0, QTableWidgetItem("检测耗时"))
        self.table_detailed_info.setItem(8, 0, QTableWidgetItem("预处理方法"))
        self.table_detailed_info.setItem(9, 0, QTableWidgetItem("颜色校准"))
        self.table_detailed_info.setItem(10, 0, QTableWidgetItem("置信度阈值"))
        self.table_detailed_info.setItem(11, 0, QTableWidgetItem("文件名"))
    
    def apply_theme(self):
        """应用主题颜色"""
        app = QApplication.instance()
        app.setStyle('Fusion')
        
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(245, 245, 245))
        palette.setColor(QPalette.WindowText, QColor(33, 33, 33))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(33, 33, 33))
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(33, 33, 33))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(76, 175, 80))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        
        app.setPalette(palette)
    
    def open_image(self):
        """打开图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择图片", 
            "", 
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.current_image = self.processor.load_image(file_path)
            
            if self.current_image is not None:
                # 更新UI显示
                self.update_image_display()
                
                # 启用检测按钮
                self.btn_start_detection.setEnabled(True)
                
                # 更新状态信息
                file_size = os.path.getsize(file_path) / 1024  # KB
                height, width = self.current_image.shape[:2]
                self.label_current_image_info.setText(f"当前图片: {width}x{height}, {file_size:.1f}KB")
                
                # 更新表格信息
                self.table_detailed_info.setItem(11, 1, QTableWidgetItem(os.path.basename(file_path)))
                self.table_detailed_info.setItem(1, 1, QTableWidgetItem(f"{width}x{height}"))
                self.table_detailed_info.setItem(2, 1, QTableWidgetItem(f"{file_size:.1f}KB"))
                
                # 更新状态标签
                self.status_label.setText(f"已加载图片: {os.path.basename(file_path)}，可以开始检测")
    
    def update_image_display(self):
        """更新图像显示"""
        if self.current_image is not None:
            # 转换图像格式
            height, width, channel = self.current_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.current_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 缩放以适应显示区域
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                self.label_original_image.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # 更新原始图像显示
            self.label_original_image.setPixmap(scaled_pixmap)
            self.label_original_image.setText("")
    
    def start_detection(self):
        """开始检测"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图片！")
            return
        
        # 禁用开始按钮，防止重复点击
        self.btn_start_detection.setEnabled(False)
        self.btn_open_image.setEnabled(False)
        
        # 重置进度条
        self.progress_bar.setValue(0)
        self.status_label.setText("正在检测...")
        
        # 创建检测线程
        use_calibration = self.check_enable_calibration.isChecked()
        self.detection_thread = DetectionThread(
            self.processor, 
            self.predictor, 
            self.current_image_path, 
            use_calibration
        )
        
        # 连接信号
        self.detection_thread.progress_update.connect(self.on_detection_progress)
        self.detection_thread.detection_complete.connect(self.on_detection_complete)
        self.detection_thread.detection_error.connect(self.on_detection_error)
        
        # 启动线程
        self.detection_thread.start()
    
    def on_detection_progress(self, progress: int, message: str):
        """检测进度更新"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"检测中: {message} ({progress}%)")
    
    def on_detection_complete(self, predictions: dict):
        """检测完成"""
        # 保存预测结果
        self.current_predictions = predictions
        
        # 更新UI显示
        self.update_result_display(predictions)
        
        # 更新处理后图像显示
        self.update_processed_image_display()
        
        # 更新统计信息
        self.processed_images_count += 1
        self.label_processed_count.setText(f"已处理图片: {self.processed_images_count}")
        
        # 启用保存按钮
        self.btn_save_result.setEnabled(True)
        self.btn_export_results.setEnabled(True)
        
        # 启用开始按钮
        self.btn_start_detection.setEnabled(True)
        self.btn_open_image.setEnabled(True)
        
        # 更新状态
        self.status_label.setText("检测完成！")
        
        # 更新时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.table_detailed_info.setItem(0, 1, QTableWidgetItem(current_time))
    
    def on_detection_error(self, error_message: str):
        """检测错误"""
        QMessageBox.critical(self, "检测错误", f"检测过程中出现错误：{error_message}")
        
        # 恢复按钮状态
        self.btn_start_detection.setEnabled(True)
        self.btn_open_image.setEnabled(True)
        
        # 更新状态
        self.status_label.setText(f"检测失败: {error_message}")
    
    def update_result_display(self, predictions: dict):
        """更新结果显示"""
        try:
            # 提取预测结果
            tf_value, tf_confidence = predictions.get('tf', (0.0, 0.0))
            tr_value, tr_confidence = predictions.get('tr', (0.0, 0.0))
            tb_value, tb_confidence = predictions.get('tb', (0.0, 0.0))
            
            # 更新数值标签
            self.label_tf_result.setText(f"{tf_value:.3f} %")
            self.label_tr_result.setText(f"{tr_value:.3f} %")
            self.label_tb_result.setText(f"{tb_value:.3f} %")
            
            # 计算总色素含量
            total_pigments = tf_value + tr_value + tb_value
            self.label_total_pigments.setText(f"总色素含量: {total_pigments:.3f} %")
            
            # 更新详细信息表格
            self.table_detailed_info.setItem(3, 1, QTableWidgetItem(f"{tf_confidence:.3f}"))
            self.table_detailed_info.setItem(4, 1, QTableWidgetItem(f"{tr_confidence:.3f}"))
            self.table_detailed_info.setItem(5, 1, QTableWidgetItem(f"{tb_confidence:.3f}"))
            
            # 计算平均置信度
            avg_confidence = (tf_confidence + tr_confidence + tb_confidence) / 3
            self.table_detailed_info.setItem(6, 1, QTableWidgetItem(f"{avg_confidence:.3f}"))
            
            # 更新预处理方法和校准状态
            enhancement = self.combo_enhancement.currentText()
            filter_method = self.combo_filter.currentText()
            color_space = self.combo_color_space.currentText()
            
            preprocess_method = f"增强:{enhancement}, 滤波:{filter_method}, 颜色空间:{color_space}"
            self.table_detailed_info.setItem(8, 1, QTableWidgetItem(preprocess_method))
            
            calibration_status = "已启用" if self.check_enable_calibration.isChecked() else "未启用"
            self.table_detailed_info.setItem(9, 1, QTableWidgetItem(calibration_status))
            
            confidence_threshold = self.slider_confidence.value() / 100
            self.table_detailed_info.setItem(10, 1, QTableWidgetItem(f"{confidence_threshold:.2f}"))
            
            # 更新可视化图表
            self.visualization_widget.update_results(predictions)
            
        except Exception as e:
            print(f"结果显示更新失败: {e}")
    
    def update_processed_image_display(self):
        """更新处理后图像显示"""
        if self.processed_image is not None:
            # 转换图像格式
            height, width, channel = self.processed_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.processed_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 缩放以适应显示区域
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                self.label_processed_image.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # 更新处理后图像显示
            self.label_processed_image.setPixmap(scaled_pixmap)
            self.label_processed_image.setText("")
            
            # 自动切换到处理后图像标签页
            self.tab_widget.setCurrentIndex(1)
    
    def save_result(self):
        """保存单次检测结果"""
        if self.current_predictions is None:
            QMessageBox.warning(self, "警告", "没有可保存的结果！")
            return
        
        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存检测结果",
            "",
            "CSV文件 (*.csv);;文本文件 (*.txt);;JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                success = self.predictor.save_predictions(self.current_predictions, file_path)
                if success:
                    QMessageBox.information(self, "成功", "检测结果已保存！")
                    self.status_label.setText(f"结果已保存到: {file_path}")
                else:
                    QMessageBox.warning(self, "警告", "保存失败！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存过程中出现错误：{e}")
    
    def export_results(self):
        """导出批量检测结果"""
        # 这里可以实现批量检测结果的导出功能
        QMessageBox.information(self, "信息", "批量导出功能正在开发中...")
    
    def on_confidence_changed(self, value: int):
        """置信度滑块变化"""
        self.label_confidence_value.setText(f"{value}%")
    
    def on_calibration_checkbox_changed(self, state: int):
        """校准复选框状态变化"""
        enabled = (state == Qt.Checked)
        self.btn_select_color_card.setEnabled(enabled)
        self.btn_perform_calibration.setEnabled(enabled)
    
    def select_color_card(self):
        """选择标准色卡"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标准色卡图片",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            # 这里可以实现色卡加载逻辑
            QMessageBox.information(self, "信息", f"已选择色卡：{file_path}")
    
    def perform_calibration(self):
        """执行颜色校准"""
        # 这里可以实现颜色校准逻辑
        QMessageBox.information(self, "信息", "颜色校准功能正在开发中...")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()