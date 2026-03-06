# -*- coding: utf-8 -*-
"""
茶色素智能检测系统 - 配置文件
Tea Pigment Intelligent Detection System - Configuration
"""

import os


class Config:
    """系统配置类"""
    
    # ============ 基础路径配置 ============
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 数据目录
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    TEMP_DIR = os.path.join(BASE_DIR, 'temp')
    
    # 模型文件路径
    MODEL_FILE = os.path.join(MODEL_DIR, 'tea_pigment_model.pkl')
    MODEL_METADATA = os.path.join(MODEL_DIR, 'model_metadata.json')
    
    # ============ 图像处理配置 ============
    # 支持的图像格式
    SUPPORTED_IMAGE_FORMATS = [
        '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp'
    ]
    
    # 图像预处理参数
    IMAGE_RESIZE_WIDTH = 800
    IMAGE_RESIZE_HEIGHT = 600
    
    # 颜色提取区域（从图像中心提取，避免边缘干扰）
    COLOR_REGION_RATIO = 0.6  # 提取中心60%区域
    
    # ============ 颜色特征配置 ============
    # 提取的颜色空间
    COLOR_SPACES = ['RGB', 'HSV', 'LAB', 'YCrCb']
    
    # ============ 模型预测配置 ============
    # 默认置信度阈值
    DEFAULT_CONFIDENCE_THRESHOLD = 0.95
    
    # 茶色素含量范围（用于验证）
    TF_RANGE = (0.0, 2.0)    # 茶黄素含量范围 (%)
    TR_RANGE = (0.0, 15.0)   # 茶红素含量范围 (%)
    TB_RANGE = (0.0, 20.0)   # 茶褐素含量范围 (%)
    
    # ============ UI配置 ============
    # 窗口默认尺寸
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 800
    
    # 界面主题
    THEME_DARK = True
    
    # ============ 报告配置 ============
    # 报告输出格式
    REPORT_FORMATS = ['CSV', 'JSON', 'Excel']
    
    # ============ 调试配置 ============
    DEBUG_MODE = False
    
    @classmethod
    def ensure_directories(cls):
        """确保所有必要的目录存在"""
        directories = [
            cls.DATA_DIR,
            cls.MODEL_DIR,
            cls.OUTPUT_DIR,
            cls.TEMP_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_model_info(cls):
        """获取模型信息（如果存在）"""
        import json
        if os.path.exists(cls.MODEL_METADATA):
            with open(cls.MODEL_METADATA, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
